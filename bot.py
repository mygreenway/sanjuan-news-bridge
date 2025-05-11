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

# Логи
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

CACHE_FILE = "titles_cache.json"
recent_summaries = []

# --- Кеш ---
def load_published_titles():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return set(json.load(f))
    return set()

def save_published_titles(titles):
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(list(titles), f, ensure_ascii=False, indent=2)

published_titles = load_published_titles()

# --- Статья ---
def get_full_article(url):
    downloaded = trafilatura.fetch_url(url)
    return trafilatura.extract(downloaded) if downloaded else ""

# --- Резюме (финальное) ---
async def improve_summary_with_gpt(title, full_article, link):
    prompt = (
        f"Escribe una publicación para Telegram sobre la siguiente noticia. Sigue este formato exacto:\n\n"
        f"1. En la primera línea, escribe el título precedido por un emoji temático y la bandera del país. Usa formato HTML así: <b>⚡ 🇪🇸 Título</b>\n"
        f"2. En un párrafo aparte, resume la noticia en 1 o 2 frases (máx. 400 caracteres). Inserta el enlace <a href=\"{link}\">en una palabra clave</a>, pero nunca al final del párrafo.\n"
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
        text = response.choices[0].message.content.strip()[:1000]

        # Автофильтр: перенос ссылки из конца
        if text.endswith("</a>"):
            match = re.search(r'<a href="[^"]+">[^<]+</a>', text)
            if match:
                link_html = match.group(0)
                text = text.replace(link_html, '')
                dot_pos = text.find('.')
                insert_pos = dot_pos + 1 if dot_pos != -1 else len(text) // 2
                text = text[:insert_pos] + ' ' + link_html + ' ' + text[insert_pos:]

        return text
    except Exception as e:
        logging.error(f"GPT error (resumen): {e}")
        return full_article[:400]

# --- Проверка на дубли ---
async def is_new_meaningful(candidate_summary, recent_summaries):
    joined = "\n".join(f"- {s}" for s in recent_summaries)
    prompt = (
        f"Estás ayudando a un bot de noticias en Telegram a evitar repetir el mismo contenido.\n\n"
        f"Últimos resúmenes publicados:\n{joined}\n\n"
        f"Nuevo resumen candidato:\n{candidate_summary}\n\n"
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
        return response.choices[0].message.content.strip().lower() == "nueva"
    except Exception as e:
        logging.error(f"GPT error (comparación): {e}")
        return True

# --- Основной цикл ---
async def fetch_and_publish():
    global published_titles
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

            full_article = get_full_article(link)
            if not full_article:
                full_article = summary

            # Предварительное резюме
            short_prompt = f"Resume brevemente esta noticia (máx. 400 caracteres):\n\nTítulo: {title}\n\nTexto:\n{full_article[:1500]}"
            try:
                short_response = await openai.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": short_prompt}],
                    temperature=0.3,
                    max_tokens=200
                )
                draft_summary = short_response.choices[0].message.content.strip()
            except Exception as e:
                logging.error(f"GPT error (resumen preliminar): {e}")
                draft_summary = full_article[:400]

            is_new = await is_new_meaningful(draft_summary, recent_summaries)
            if not is_new:
                logging.info("⏩ Noticia repetida por sentido. Se omite.")
                continue

            improved_text = await improve_summary_with_gpt(title, full_article, link)
            recent_summaries.append(draft_summary)
            if len(recent_summaries) > 10:
                recent_summaries.pop(0)

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

            try:
                for channel in CHANNEL_IDS:
                    if image_url:
                        await bot.send_photo(chat_id=channel, photo=image_url, caption=improved_text, parse_mode=ParseMode.HTML)
                    else:
                        await bot.send_message(chat_id=channel, text=improved_text, parse_mode=ParseMode.HTML)
                published_titles.add(title_key)
                save_published_titles(published_titles)
                await asyncio.sleep(5)
            except Exception as e:
                logging.error(f"❌ Telegram error en {channel}: {e}")

# --- Цикл ---
async def main_loop():
    while True:
        logging.info("🔄 Comprobando noticias...")
        await fetch_and_publish()
        await asyncio.sleep(1800)

if __name__ == "__main__":
    asyncio.run(main_loop())
