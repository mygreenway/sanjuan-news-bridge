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

async def notify_admin(message):
    try:
        await bot.send_message(chat_id=ADMIN_CHAT_ID, text=message)
    except Exception as e:
        logging.error(f"Error notifying admin: {e}")

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

async def improve_summary_with_gpt(title, full_article, link):
    prompt = (
        f"Escribe una publicación para Telegram sobre la siguiente noticia. Sigue exactamente este formato:\n\n"
        f"1. Una primera línea con un emoji temático y el título en negrita usando <b> ... </b>.\n"
        f"2. Luego un párrafo separado (máx. 400 caracteres) que resuma la noticia con claridad. No uses frases como 'más información en el enlace' o similares. El texto debe tener valor completo por sí mismo.\n"
        f"3. Dentro del párrafo, inserta el enlace <a href=\"{link}\">en una palabra clave</a> si es posible, nunca al final.\n"
        f"4. Una línea final con 2 o 3 hashtags relevantes.\n\n"
        f"Título: {title}\n\nTexto:\n{full_article[:2000]}"
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

            draft_summary = full_article[:400]
            is_new = await is_new_meaningful(draft_summary, recent_summaries)
            if not is_new:
                continue

            improved_text = await improve_summary_with_gpt(title, full_article, entry.link)
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
