import os
import asyncio
import feedparser
import openai
import requests
from telegram import Bot
from telegram.constants import ParseMode

# Ключ OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")

# Телеграм бот
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHANNEL_ID = "@sanjuan_online"
bot = Bot(token=BOT_TOKEN)

# RSS-источники
RSS_FEEDS = [
    "https://e00-elmundo.uecdn.es/elmundo/rss/portada.xml",
    "https://feeds.elpais.com/mrss-s/pages/ep/site/elpais.com/portada",
    "https://www.rtve.es/rss/portal/rss.xml",
    "https://www.20minutos.es/rss/",
    "https://www.europapress.es/rss/rss.aspx"
]

published_titles = set()

async def mejorar_texto_con_gpt(texto_original):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # Или "gpt-4o", если доступен
            messages=[
                {"role": "system", "content": "Eres un redactor profesional de noticias. Resume esta noticia en español de forma clara, completa y atractiva."},
                {"role": "user", "content": texto_original}
            ],
            max_tokens=600,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print("GPT error:", e)
        return texto_original

async def fetch_and_publish():
    for url in RSS_FEEDS:
        feed = feedparser.parse(url)
        for entry in feed.entries[:1]:  # Только одна новость из источника
            title = entry.get("title", "")
            link = entry.get("link", "")
            raw_summary = entry.get("summary", "")[:1000]
            image_url = ""

            if title in published_titles:
                continue

            # Обработка изображения
            if "media_content" in entry:
                image_url = entry.media_content[0].get("url", "")
            elif "image" in entry and isinstance(entry.image, dict):
                image_url = entry.image.get("href", "")

            # Генерируем улучшенное описание
            resumen = await mejorar_texto_con_gpt(raw_summary)

            # Оформление поста
            hashtags = "#Noticias #España #SanJuan"
            mensaje = f"<b>{title}</b>\n\n{resumen}\n\n<a href='{link}'>Leer más</a>\n\n{hashtags}"

            # Публикация
            try:
                if image_url:
                    await bot.send_photo(
                        chat_id=CHANNEL_ID,
                        photo=image_url,
                        caption=mensaje,
                        parse_mode=ParseMode.HTML
                    )
                else:
                    await bot.send_message(
                        chat_id=CHANNEL_ID,
                        text=mensaje,
                        parse_mode=ParseMode.HTML
                    )
                published_titles.add(title)
                await asyncio.sleep(5)
            except Exception as e:
                print("Telegram error:", e)

async def main_loop():
    while True:
        print("🔄 Проверка новостей...")
        await fetch_and_publish()
        await asyncio.sleep(1800)

if __name__ == "__main__":
    asyncio.run(main_loop())
