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
recent_summaries = []
MAX_PUBLICATIONS_PER_CYCLE = 5

def load_published_titles():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return set(json.load(f))
    return set()

def save_published_titles(titles):
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(list(titles), f, ensure_ascii=False, indent=2)

published_titles = load_published_titles()

def get_full_article(url):
    downloaded = trafilatura.fetch_url(url)
    return trafilatura.extract(downloaded) if downloaded else ""

async def notify_admin(message):
    try:
        await bot.send_message(chat_id=ADMIN_CHAT_ID, text=message)
    except Exception as e:
        logging.error(f"Error notifying admin: {e}")

async def improve_summary_with_gpt(title, full_article, link):
    max_length = 2000 if any(word in link.lower() for word in ["opinion", "analis", "editorial", "tribuna"]) else 1500
    trimmed_article = full_article[:max_length]

    prompt = (
        f"Escribe una publicación para Telegram sobre la siguiente noticia. Sigue este formato ESTRICTAMENTE:\n\n"
        f"1. Una primera línea con un emoji temático y el título en negrita usando <b> ... </b> (NO uses **).\n"
        f"2. Luego un párrafo (máx. 400 caracteres) que resuma claramente la noticia. No repitas el título.\n"
        f"3. No añadas ningún enlace. Sólo texto.\n"
        f"4. Una última línea con 2 o 3 hashtags relevantes, separados por espacios.\n\n"
        f"Título: {title}\n\nTexto:\n{trimmed_article}"
    )

    response = await openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.6,
        max_tokens=400
    )
    return response.choices[0].message.content.strip()[:1000]

async def is_new_meaningful(candidate_summary, recent_summaries):
    joined = "\n".join(f"- {s}" for s in recent_summaries)
    prompt = f"¿El resumen es nuevo o repetido?\n\nÚltimos:\n{joined}\n\nNuevo:\n{candidate_summary}\n\nResponde 'nueva' o 'repetida'."
    response = await openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=10
    )
    return response.choices[0].message.content.strip().lower() == "nueva"

async def fetch_and_publish():
    global published_titles
    published_count = 0

    for url in RSS_FEEDS:
        feed = feedparser.parse(url)
        for entry in feed.entries[:1]:
            if published_count >= MAX_PUBLICATIONS_PER_CYCLE:
                return

            title = re.sub(r'^[^:|]+[|:]', '', entry.title).strip()
            title_key = re.sub(r'\W+', '', title.lower())

            if title_key in published_titles:
                continue

            full_article = get_full_article(entry.link) or entry.summary
            if len(full_article.split()) < 80:
                continue

            draft_summary = full_article[:400]
            is_new = await is_new_meaningful(draft_summary, recent_summaries)
            if not is_new:
                continue

            improved_text = await improve_summary_with_gpt(title, full_article, entry.link)

            # Вставка ссылки вручную в 1-2 слово основного текста
            link_html = f'<a href="{entry.link}">'
            closing_tag = '</a>'
            lines = improved_text.split("\n")
            if len(lines) > 1:
                words = lines[1].split()
                if len(words) >= 2:
                    words[1] = f'{link_html}{words[1]}{closing_tag}'
                elif words:
                    words[0] = f'{link_html}{words[0]}{closing_tag}'
                lines[1] = " ".join(words)
                improved_text = "\n".join(lines)

            # Добавить подпись с кликабельной ссылкой на канал
            improved_text += '\n\n<a href="https://t.me/NoticiasEspanaHoy">📡 Noticias de España</a>'

            recent_summaries.append(draft_summary)
            recent_summaries[:] = recent_summaries[-10:]

            image_url = entry.media_content[0]["url"] if "media_content" in entry else ""

            try:
                for channel in CHANNEL_IDS:
                    if image_url:
                        await bot.send_photo(channel, image_url, caption=improved_text, parse_mode=ParseMode.HTML)
                    else:
                        await bot.send_message(channel, improved_text, parse_mode=ParseMode.HTML, disable_web_page_preview=True)
                published_titles.add(title_key)
                save_published_titles(published_titles)
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
