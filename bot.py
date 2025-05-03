import os
import asyncio
import feedparser
import requests
from telegram import Bot
from telegram.constants import ParseMode
from openai import OpenAI
from datetime import datetime
from html import escape

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHANNEL_ID = "@sanjuan_online"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

bot = Bot(token=BOT_TOKEN)
client = OpenAI(api_key=OPENAI_API_KEY)

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
    Eres un redactor profesional para un canal de noticias en Telegram dirigido a una audiencia española.
    Tu tarea es redactar una publicación clara, profesional y atractiva basada en esta noticia.

    Requisitos:
    - Comienza con un título impactante sin usar etiquetas como "Título:"
    - Escribe un resumen expandido, objetivo y bien estructurado
    - Mantén un tono informativo y periodístico
    - Añade emojis solo si son apropiados y relevantes
    - Inserta el enlace al final de una frase como "Leer más" usando formato Markdown: [Leer más]({link})
    - No muestres la URL completa en el texto
    - Escribe únicamente en español

    Noticia:
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

            if title in published_titles:
                continue

            formatted_post = await generate_post_with_gpt(title, summary, link)
            
            if not formatted_post:
                continue

            try:
                await bot.send_message(
                    chat_id=CHANNEL_ID,
                    text=f"{formatted_post}\n\n#Noticias #España #SanJuan",
                    parse_mode=ParseMode.MARKDOWN,
                    disable_web_page_preview=False
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
