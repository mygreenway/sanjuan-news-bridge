import os
import asyncio
import feedparser
from telegram import Bot
from telegram.constants import ParseMode
from openai import OpenAI

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHANNEL_ID = "@sanjuan_online"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

bot = Bot(token=BOT_TOKEN)
client = OpenAI(api_key=OPENAI_API_KEY)

# RSS источники: крупные испанские издания и Twitter-политики
RSS_FEEDS = [
    "https://feeds.elpais.com/mrss-s/pages/ep/site/elpais.com/portada",
    "https://e00-elmundo.uecdn.es/elmundo/rss/portada.xml",
    "https://www.rtve.es/rss/portal/rss.xml",
    "https://www.20minutos.es/rss/",
    "https://www.lavanguardia.com/mvc/feed/rss/home",
    "https://www.eldiario.es/rss/",
    "https://www.abc.es/rss/feeds/abcPortada.xml",
    "https://www.larazon.es/rss/",
    "https://okdiario.com/feed",
    "https://www.publico.es/rss",
    "https://www.europapress.es/rss/rss.aspx",
    "https://cadenaser.com/feed/",
    "https://www.cope.es/rss/rss.xml",
    "https://www.elconfidencial.com/rss/ultimas_noticias.xml",
]

TWITTER_RSS = [
    "https://rsshub.app/twitter/user/sanchezcastejon",
    "https://rsshub.app/twitter/user/Yolanda_Diaz_",
    "https://rsshub.app/twitter/user/AlbertoNunezFeijoo",
    "https://rsshub.app/twitter/user/KingFelipeVI",
]

published_titles = set()

async def generate_post_with_gpt(title, summary, link):
    prompt = f"""
Eres un redactor profesional de noticias en español para Telegram. Basado en el siguiente titular, resumen y enlace, redacta una publicación clara, estructurada y atractiva para un canal informativo.

Requisitos:
- Encabezado con un emoji relevante + título en negrita.
- 1–2 párrafos informativos.
- Enlace oculto в виде [Leer más](URL).
- Без фраз вроде "Título:", "Resumen:", просто оформленный пост.
- В конце хештеги: #Noticias #España #SanJuan
- Пиши только по-испански. Без формальностей.

Título: {title}
Resumen: {summary}
Enlace: {link}
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Eres un redactor profesional de noticias para Telegram."},
                {"role": "user", "content": prompt},
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print("❌ GPT error:", e)
        return None

async def fetch_and_publish():
    for url in RSS_FEEDS + TWITTER_RSS:
        feed = feedparser.parse(url)
        for entry in feed.entries[:1]:
            title = entry.get("title", "")
            link = entry.get("link", "")
            summary = entry.get("summary", "")[:500]
            image_url = ""

            if title in published_titles:
                continue

            if "media_content" in entry:
                image_url = entry.media_content[0]["url"]
            elif "image" in entry:
                image_url = entry.image.get("href", "")

            formatted_post = await generate_post_with_gpt(title, summary, link)
            if not formatted_post:
                continue

            try:
                if image_url:
                    await bot.send_photo(
                        chat_id=CHANNEL_ID,
                        photo=image_url,
                        caption=formatted_post,
                        parse_mode=ParseMode.MARKDOWN
                    )
                else:
                    await bot.send_message(
                        chat_id=CHANNEL_ID,
                        text=formatted_post,
                        parse_mode=ParseMode.MARKDOWN
                    )

                published_titles.add(title)
                await asyncio.sleep(5)
            except Exception as e:
                print("Telegram error:", e)

async def main_loop():
    while True:
        print("🔄 Buscando noticias...")
        await fetch_and_publish()
        await asyncio.sleep(1800)

if __name__ == "__main__":
    asyncio.run(main_loop())
