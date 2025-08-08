import os
import re
import json
import asyncio
import feedparser
from telegram import Bot
from telegram.constants import ParseMode
from openai import AsyncOpenAI
import trafilatura
import logging
from itertools import tee

logging.basicConfig(
    filename='bot_log.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CHANNEL_IDS = ["@NoticiasEspanaHoy"]
ADMIN_CHAT_ID = os.getenv("ADMIN_CHAT_ID")

bot = Bot(token=BOT_TOKEN)
openai = AsyncOpenAI(api_key=OPENAI_API_KEY)

RSS_FEEDS = [
    "https://e00-elmundo.uecdn.es/elmundo/rss/portada.xml",
    "https://feeds.elpais.com/mrss-s/pages/ep/site/elpais.com/portada",
    "https://www.rtve.es/rss/portal/rss.xml",
    "https://www.20minutos.es/rss/",
    "https://www.europapress.es/rss/rss.aspx",
    "https://www.abc.es/rss/feeds/abcPortada.xml",
    "https://www.lavanguardia.com/mvc/feed/rss/home",
    "https://www.elconfidencial.com/rss/espana.xml",
    "https://www.eldiario.es/rss/",
    "https://www.publico.es/rss/",
    "https://www.lasprovincias.es/rss/2.0/portada/index.rss"
]

CACHE_FILE = "titles_cache.json"
MAX_PUBLICATIONS_PER_CYCLE = 5
MAX_CACHE = 500

# --------------------- Утилиты ---------------------

def normalize_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()

def bigrams(tokens):
    a, b = tee(tokens)
    next(b, None)
    return list(zip(a, b))

def jaccard_similarity(a, b):
    set_a, set_b = set(a), set(b)
    if not set_a or not set_b:
        return 0
    return len(set_a & set_b) / len(set_a | set_b)

def is_duplicate(title, content, history):
    """Проверка на дубликат по заголовку и тексту"""
    norm_title = normalize_text(title)
    title_tokens = norm_title.split()
    title_bigrams = bigrams(title_tokens)

    content = normalize_text(content)
    content_tokens = content.split()[:300]
    content_bigrams = bigrams(content_tokens)

    for old in history:
        # Проверка по заголовку
        old_title_bigrams = bigrams(old["title"].split())
        if jaccard_similarity(title_bigrams, old_title_bigrams) >= 0.85:
            return True

        # Проверка по тексту
        old_content_bigrams = bigrams(old["content"].split())
        if jaccard_similarity(content_bigrams, old_content_bigrams) >= 0.90:
            return True

    return False

def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_cache(cache):
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache[-MAX_CACHE:], f, ensure_ascii=False, indent=2)

def get_full_article(url):
    downloaded = trafilatura.fetch_url(url)
    return trafilatura.extract(downloaded) if downloaded else ""

async def notify_admin(message):
    try:
        await bot.send_message(chat_id=ADMIN_CHAT_ID, text=message)
    except Exception as e:
        logging.error(f"Error notifying admin: {e}")

# --------------------- GPT: Генерация поста ---------------------

async def improve_summary_with_gpt(title, full_article, link):
    trimmed_article = full_article[:1500]

    prompt = (
        f"Escribe una publicación para Telegram sobre la siguiente noticia. Sigue este formato ESTRICTAMENTE:\n\n"
        f"1. Una primera línea con un emoji temático y el título en negrita usando <b> ... </b>.\n"
        f"2. Luego un párrafo (máx. 400 caracteres) que resuma claramente la noticia. No repitas el título.\n"
        f"3. Inserta el enlace {link} en UNA sola palabra clave del resumen (no en el título).\n"
        f"4. Una última línea con 2 o 3 hashtags relevantes.\n\n"
        f"Título: {title}\n\nTexto:\n{trimmed_article}"
    )

    response = await openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.6,
        max_tokens=400
    )
    return response.choices[0].message.content.strip()

# --------------------- Основная логика ---------------------

async def fetch_and_publish():
    cache = load_cache()
    published_count = 0

    for url in RSS_FEEDS:
        if published_count >= MAX_PUBLICATIONS_PER_CYCLE:
            break

        feed = feedparser.parse(url)
        for entry in feed.entries[:2]:
            if published_count >= MAX_PUBLICATIONS_PER_CYCLE:
                break

            title = re.sub(r'^[^:|]+[|:]', '', entry.title).strip()
            full_article = get_full_article(entry.link) or entry.summary

            if len(full_article.split()) < 50:
                continue

            if is_duplicate(title, full_article, cache):
                continue

            improved_text = await improve_summary_with_gpt(title, full_article, entry.link)

            # Добавляем ссылку на канал в конце
            improved_text += "\n\n<a href='https://t.me/NoticiasEspanaHoy'>📡 Noticias de España</a>"

            image_url = entry.media_content[0]["url"] if "media_content" in entry else ""

            try:
                for channel in CHANNEL_IDS:
                    if image_url:
                        await bot.send_photo(channel, image_url, caption=improved_text, parse_mode=ParseMode.HTML)
                    else:
                        await bot.send_message(channel, improved_text, parse_mode=ParseMode.HTML, disable_web_page_preview=False)

                cache.append({
                    "title": normalize_text(title),
                    "content": normalize_text(" ".join(full_article.split()[:300]))
                })
                save_cache(cache)
                published_count += 1
                await asyncio.sleep(5)
            except Exception as e:
                logging.error(f"Telegram error: {e}")
                await notify_admin(f"❌ Error publicación: {e}")

async def main_loop():
    while True:
        logging.info("🔄 Comprobando noticias...")
        await fetch_and_publish()
        await asyncio.sleep(1800)

if __name__ == "__main__":
    asyncio.run(main_loop())
