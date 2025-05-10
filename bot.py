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
CHANNEL_IDS = ["@sanjuan_online", "@NoticiasEspanaHoy"]

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
        f"Crea una publicación profesional para Telegram con este formato:\n"
        f"1. Primera línea: emoji temático, banderas relevantes y título (sin la palabra 'Título'). Formato HTML: <b>⚡ 🇺🇸🇪🇺 Aquí título claro</b>\n"
        f"2. Segundo párrafo: resume completamente la noticia (máximo 400 caracteres), el lector debe entender toda la noticia SIN abrir enlaces externos. Inserta naturalmente un enlace HTML (<a href=\"{link}\">palabra clave</a>).\n"
        f"3. Finaliza con 2-3 hashtags populares en español.\n"
        f"No agregues frases adicionales. Sé neutral e informativo.\n\n"
        f"Título: {title}\nTexto: {full_article[:1500]}"
    )
    for _ in range(2):
        try:
            response = await openai.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=400
            )
            content = response.choices[0].message.content.strip()
            content = re.sub(r'```html|```|Título:', '', content).strip()
            return content[:1000]
        except Exception as e:
            logging.error(f"GPT error (resumen): {e}")
            await asyncio.sleep(3)
    return full_article[:350]

async def is_new_meaningful(improved_text, recent_summaries):
    joined = "\n".join(recent_summaries)
    prompt = (
        f"Últimas noticias:\n{joined}\n\n"
        f"Nueva noticia:\n{improved_text}\n\n"
        f"Si es distinta responde SOLO 'nueva', si no, responde 'repetida'."
    )
    try:
        response = await openai.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=5
        )
        return response.choices[0].message.content.strip().lower() == "nueva"
    except Exception as e:
        logging.error(f"GPT error (comparación): {e}")
        return True

async def fetch_and_publish():
    for url in RSS_FEEDS:
        feed = feedparser.parse(url)
        for entry in feed.entries[:1]:
            title = re.sub(r'^[^:|]+[|:]\s*', '', entry.get("title", "")).strip()
            link = entry.get("link", "")
            summary = entry.get("summary", "")

            title_key = re.sub(r'[^\w\s]', '', title.lower()).strip()
            if title_key in published_titles:
                continue
            published_titles.add(title_key)

            image_url = next((entry.get(key)[0]["url"] for key in ["media_content", "media_thumbnail"] if key in entry), "")
            if not image_url:
                img_match = re.search(r'<img[^>]+src="([^">]+)"', summary)
                image_url = img_match.group(1) if img_match else ""

            full_article = get_full_article(link) or summary
            improved_text = await improve_summary_with_gpt(title, full_article, link)

            if not await is_new_meaningful(improved_text, recent_summaries):
                logging.info(f"Noticia repetida: {title}")
                continue

            recent_summaries.append(improved_text)
            if len(recent_summaries) > 10:
                recent_summaries.pop(0)

            for channel in CHANNEL_IDS:
                try:
                    if image_url:
                        await bot.send_photo(chat_id=channel, photo=image_url, caption=improved_text, parse_mode=ParseMode.HTML)
                    else:
                        await bot.send_message(chat_id=channel, text=improved_text, parse_mode=ParseMode.HTML)
                    logging.info(f"Publicado en {channel}: {title}")
                    await asyncio.sleep(5)
                except Exception as e:
                    logging.error(f"Telegram error {channel}: {e}")

async def main_loop():
    while True:
        logging.info("Buscando nuevas noticias...")
        await fetch_and_publish()
        await asyncio.sleep(1800)

if __name__ == "__main__":
    asyncio.run(main_loop())
