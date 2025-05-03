import os
import re
import asyncio
import feedparser
from telegram import Bot
from telegram.constants import ParseMode
from openai import AsyncOpenAI
import trafilatura

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHANNEL_ID = "@sanjuan_online"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

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

def detect_emoji(text):
    text = text.lower()
    icon = "📰"
    flag = "🇪🇸"

    if any(word in text for word in ["electricidad", "energía", "apagón", "eléctrico"]):
        icon = "⚡"
    elif any(word in text for word in ["política", "gobierno", "elecciones", "parlamento"]):
        icon = "🏛️"
    elif any(word in text for word in ["economía", "empleo", "precios", "inflación"]):
        icon = "💰"
    elif any(word in text for word in ["accidente", "incendio", "policía", "muerte", "suceso"]):
        icon = "🚨"
    elif any(word in text for word in ["lluvia", "tormenta", "clima", "temperatura", "calor"]):
        icon = "🌧️"

    if "españa" in text:
        flag = "🇪🇸"
    elif "francia" in text:
        flag = "🇫🇷"
    elif "alemania" in text:
        flag = "🇩🇪"
    elif "italia" in text:
        flag = "🇮🇹"
    elif "reino unido" in text or "gran bretaña" in text:
        flag = "🇬🇧"
    elif "eeuu" in text or "estados unidos" in text or "usa" in text:
        flag = "🇺🇸"
    elif "rusia" in text:
        flag = "🇷🇺"
    elif "ucrania" in text:
        flag = "🇺🇦"
    elif "marruecos" in text:
        flag = "🇲🇦"
    elif "china" in text:
        flag = "🇨🇳"
    elif "argentina" in text:
        flag = "🇦🇷"
    elif "méxico" in text or "mexico" in text:
        flag = "🇲🇽"

    return f"{icon} {flag}"

def get_full_article(url):
    downloaded = trafilatura.fetch_url(url)
    if downloaded:
        return trafilatura.extract(downloaded)
    return ""

async def improve_summary_with_gpt(title, full_article, link):
    prompt = (
        f"Resume esta noticia de forma muy breve y clara para una publicación en Telegram. "
        f"Limita el texto a máximo 400 caracteres. Da solo lo esencial en 1 o 2 frases. "
        f"Incorpora el siguiente enlace en una palabra clave dentro del texto: {link}.\n\n"
        f"Título: {title}\n\nTexto de la noticia:\n{full_article[:2000]}"
    )

    try:
        response = await openai.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.6,
            max_tokens=300
        )
        return response.choices[0].message.content.strip()[:1000]
    except Exception as e:
        print("GPT error:", e)
        return full_article[:400]

async def fetch_and_publish():
    for url in RSS_FEEDS:
        feed = feedparser.parse(url)
        for entry in feed.entries[:1]:
            title = entry.get("title", "")
            link = entry.get("link", "")
            summary = entry.get("summary", "")

            if title in published_titles:
                continue

            # Поиск изображения
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

            print(f"📰 Fuente: {url}")
            print(f"🔗 Imagen: {image_url or 'No encontrada'}")

            emoji = detect_emoji(title + summary)
            full_article = get_full_article(link)
            if not full_article:
                full_article = summary

            improved_text = await improve_summary_with_gpt(title, full_article, link)
            hashtags = "#Noticias #España #SanJuan"

            text = (
                f"<b>{emoji} {title}</b>\n\n"
                f"{improved_text}\n\n"
                f"{hashtags}"
            )

            try:
                if image_url:
                    await bot.send_photo(chat_id=CHANNEL_ID, photo=image_url, caption=text, parse_mode=ParseMode.HTML)
                else:
                    await bot.send_message(chat_id=CHANNEL_ID, text=text, parse_mode=ParseMode.HTML)

                published_titles.add(title)
                await asyncio.sleep(5)
            except Exception as e:
                print("❌ Telegram error:", e)

async def main_loop():
    while True:
        print("🔄 Comprobando noticias...")
        await fetch_and_publish()
        await asyncio.sleep(1800)

if __name__ == "__main__":
    asyncio.run(main_loop())
