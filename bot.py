import os
import re
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

published_titles = set()
recent_summaries = []

def get_full_article(url):
    downloaded = trafilatura.fetch_url(url)
    return trafilatura.extract(downloaded) if downloaded else ""

async def improve_summary_with_gpt(title, full_article, link):
    prompt = (
        f"Escribe una publicación para Telegram sobre la siguiente noticia. Sigue este formato exacto:\n\n"
        f"1. En la primera línea, escribe el título precedido por un emoji temático y la bandera del país. Usa formato HTML así: <b>⚡ 🇪🇸 Título</b>\n"
        f"2. En un párrafo aparte, resume la noticia en 1 o 2 frases (máx. 400 caracteres). Usa <a href=\"{link}\">palabra</a> para enlazar.\n"
        f"3. En la última línea separada, añade de 2 a 3 hashtags relevantes y populares.\n\n"
        f"Título: {title}\n\nTexto de la noticia:\n{full_article[:2000]}"
    )

    try:
        response = await openai.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.6,
            max_tokens=400
        )
        return response.choices[0].message.content.strip()[:1000]
    except Exception as e:
        logging.error(f"GPT error (resumen): {e}")
        return full_article[:400]

async def is_new_meaningful(improved_text, recent_summaries):
    joined = "\n".join(f"- {s}" for s in recent_summaries)
    prompt = (
        f"Estás ayudando a un bot de noticias en Telegram a evitar repetir el mismo contenido.\n\n"
        f"Últimos resúmenes publicados:\n{joined}\n\n"
        f"Nuevo resumen candidato:\n{improved_text}\n\n"
        f"¿Este nuevo resumen expresa una noticia realmente distinta? Si sí, responde solo con: nueva. "
        f"Si repite la misma noticia con otras palabras, responde solo con: repetida."
    )
    try:
        response = await openai.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=10
        )
        answer = response.choices[0].message.content.strip().lower()
        return answer == "nueva"
    except Exception as e:
        logging.error(f"GPT error (comparación): {e}")
        return True

async def fetch_and_publish():
    for url in RSS_FEEDS:
        feed = feedparser.parse(url)
        for entry in feed.entries[:1]:
            raw_title = entry.get("title", "")
            link = entry.get("link", "")
            summary = entry.get("summary", "")

            title = re.sub(r'^[^:|]+[|:]\s*', '', raw_title, flags=re.IGNORECASE)
            title = re.sub(r'\b(directo|última hora|en vivo)\b[:\-–—]?\s*', '', title, flags=re.IGNORECASE)

            title_key = re.sub(r'[^\w\s]', '', title.lower()).strip()
            if title_key in published_titles:
                continue
            published_titles.add(title_key)

            image_url = ""
            if "media_content" in entry:
                image_url = entry.media_content[0]["url"]
            elif "image" in entry:
                image_url = entry.image.get("href", "")
            elif "media_thumbnail" in entry:
                image_url = entry.media_thumbnail[0].get("url", "")
            elif "summary" in entry:
                match = re.search(r'<img[^>]+src="([^">]+)"', entry.summary)
                if match:
                    image_url = match.group(1)

            full_article = get_full_article(link)
            if not full_article:
                full_article = summary

            improved_text = await improve_summary_with_gpt(title, full_article, link)

            is_new = await is_new_meaningful(improved_text, recent_summaries)
            if not is_new:
                logging.info("⏩ Noticia repetida por sentido. Se omite.")
                continue

            recent_summaries.append(improved_text)
            if len(recent_summaries) > 10:
                recent_summaries.pop(0)

            try:
                for channel in CHANNEL_IDS:
                    if image_url:
                        await bot.send_photo(chat_id=channel, photo=image_url, caption=improved_text, parse_mode=ParseMode.HTML)
                    else:
                        await bot.send_message(chat_id=channel, text=improved_text, parse_mode=ParseMode.HTML)
                await asyncio.sleep(5)
            except Exception as e:
                logging.error(f"❌ Telegram error en {channel}: {e}")

async def main_loop():
    while True:
        logging.info("🔄 Comprobando noticias...")
        await fetch_and_publish()
        await asyncio.sleep(1800)

if __name__ == "__main__":
    asyncio.run(main_loop())
