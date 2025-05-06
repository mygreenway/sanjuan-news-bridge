import os
import re
import asyncio
import feedparser
from difflib import SequenceMatcher
from telegram import Bot
from telegram.constants import ParseMode
from openai import AsyncOpenAI
import trafilatura

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

def is_similar(a, b, threshold=0.85):
    return SequenceMatcher(None, a, b).ratio() >= threshold

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

    countries = {
        r"\bespaña\b": "🇪🇸",
        r"\bfrancia\b": "🇫🇷",
        r"\balemania\b": "🇩🇪",
        r"\bitalia\b": "🇮🇹",
        r"\breino unido\b": "🇬🇧",
        r"\bgran bretaña\b": "🇬🇧",
        r"\bestados unidos\b": "🇺🇸",
        r"\busa\b": "🇺🇸",
        r"\beeuu\b": "🇺🇸",
        r"\brusia\b": "🇷🇺",
        r"\bucrania\b": "🇺🇦",
        r"\bmarruecos\b": "🇲🇦",
        r"\bchina\b": "🇨🇳",
        r"\bargentina\b": "🇦🇷",
        r"\bm[ée]xico\b": "🇲🇽"
    }

    for pattern, emoji_flag in countries.items():
        if re.search(pattern, text):
            flag = emoji_flag
            break

    return f"{icon} {flag}"

def get_full_article(url):
    downloaded = trafilatura.fetch_url(url)
    if downloaded:
        return trafilatura.extract(downloaded)
    return ""

async def improve_summary_with_gpt(title, full_article, link):
    prompt = (
        f"Resume esta noticia de forma muy breve y clara para una publicación en Telegram. "
        f"Usa como máximo 400 caracteres y escribe 1 o 2 frases con la información esencial. "
        f"No uses encabezados, no repitas el título. "
        f"Incorpora el siguiente enlace en una palabra clave usando el formato HTML así: "
        f'<a href=\"{link}\">palabra</a>.\n\n'
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
            raw_title = entry.get("title", "")
            link = entry.get("link", "")
            summary = entry.get("summary", "")

            # Очистка заголовка
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

            emoji = detect_emoji(title + summary + full_article)
            improved_text = await improve_summary_with_gpt(title, full_article, link)

            # Умная проверка на дубли по смыслу
            if any(is_similar(improved_text.lower(), s) for s in recent_summaries):
                print("⏩ Noticia duplicada por similitud. Se omite.")
                continue

            recent_summaries.append(improved_text.lower())
            if len(recent_summaries) > 10:
                recent_summaries.pop(0)

            hashtags = "#Noticias #España #Actualidad"
            text = (
                f"<b>{emoji} {title}</b>\n\n"
                f"{improved_text}\n\n"
                f"{hashtags}"
            )

            try:
                for channel in CHANNEL_IDS:
                    if image_url:
                        await bot.send_photo(chat_id=channel, photo=image_url, caption=text, parse_mode=ParseMode.HTML)
                    else:
                        await bot.send_message(chat_id=channel, text=text, parse_mode=ParseMode.HTML)
                await asyncio.sleep(5)
            except Exception as e:
                print(f"❌ Telegram error en {channel}:", e)

async def main_loop():
    while True:
        print("🔄 Comprobando noticias...")
        await fetch_and_publish()
        await asyncio.sleep(1800)

if __name__ == "__main__":
    asyncio.run(main_loop())
