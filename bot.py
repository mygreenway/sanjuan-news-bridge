import os
import asyncio
import feedparser
import requests
from telegram import Bot
from telegram.constants import ParseMode
from openai import AsyncOpenAI

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHANNEL_ID = "@sanjuan_online"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

bot = Bot(token=BOT_TOKEN)
openai = AsyncOpenAI(api_key=OPENAI_API_KEY)

# RSS-источники
RSS_FEEDS = [
    "https://e00-elmundo.uecdn.es/elmundo/rss/portada.xml",
    "https://feeds.elpais.com/mrss-s/pages/ep/site/elpais.com/portada",
    "https://www.rtve.es/rss/portal/rss.xml",
    "https://www.20minutos.es/rss/",
    "https://www.europapress.es/rss/rss.aspx"
]

published_titles = set()

async def improve_summary_with_gpt(title, summary):
    prompt = (
        f"Mejora y amplía este resumen de noticia de forma clara y profesional, en español. "
        f"Asegúrate de que раскрыта основная суть новости:\n\n"
        f"Título: {title}\n\nResumen: {summary}\n\nTexto mejorado:"
    )
    try:
        response = await openai.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=400
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print("GPT error:", e)
        return summary

async def fetch_and_publish():
    for url in RSS_FEEDS:
        feed = feedparser.parse(url)
        for entry in feed.entries[:1]:
            title = entry.get("title", "")
            link = entry.get("link", "")
            summary = entry.get("summary", "")[:700]
            image_url = ""

            if title in published_titles:
                continue

            # Поиск изображения
            if "media_content" in entry:
                image_url = entry.media_content[0]["url"]
            elif "image" in entry:
                image_url = entry.image.get("href", "")

            # Улучшение текста через GPT
            improved_summary = await improve_summary_with_gpt(title, summary)

            hashtags = "#Noticias #España #SanJuan"
            text = f"<b>{title}</b>\n\n{improved_summary}\n\n<a href='{link}'>Leer más</a>\n\n{hashtags}"

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
        await asyncio.sleep(1800)  # 30 минут

if __name__ == "__main__":
    asyncio.run(main_loop())
