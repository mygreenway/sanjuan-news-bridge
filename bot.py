import os
import asyncio
import feedparser
from telegram import Bot
from telegram.constants import ParseMode

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHANNEL_ID = "@sanjuan_online"
bot = Bot(token=BOT_TOKEN)

# Популярные испанские источники (RSS)
RSS_FEEDS = [
    "https://e00-elmundo.uecdn.es/elmundo/rss/portada.xml",  # El Mundo
    "https://feeds.elpais.com/mrss-s/pages/ep/site/elpais.com/portada",  # El País
    "https://www.rtve.es/rss/portal/rss.xml",                # RTVE
    "https://www.20minutos.es/rss/",                         # 20 Minutos
    "https://www.europapress.es/rss/rss.aspx"                # Europa Press
]

# Храним уже опубликованные заголовки, чтобы не дублировать (на время жизни процесса)
published_titles = set()

async def fetch_and_publish():
    for url in RSS_FEEDS:
        feed = feedparser.parse(url)
        for entry in feed.entries[:1]:  # Только одна свежая новость из каждого источника
            title = entry.get("title", "")
            link = entry.get("link", "")
            summary = entry.get("summary", "")[:500]
            image_url = ""

            if title in published_titles:
                continue

            # Ищем изображение, если есть
            if "media_content" in entry:
                image_url = entry.media_content[0]["url"]
            elif "image" in entry:
                image_url = entry.image.get("href", "")

            hashtags = "#Noticias #España #SanJuan"
            text = f"<b>{title}</b>\n\n{summary}\n\n<a href='{link}'>Leer más</a>\n\n{hashtags}"

            try:
                if image_url:
                    await bot.send_photo(chat_id=CHANNEL_ID, photo=image_url, caption=text, parse_mode=ParseMode.HTML)
                else:
                    await bot.send_message(chat_id=CHANNEL_ID, text=text, parse_mode=ParseMode.HTML)

                published_titles.add(title)
                await asyncio.sleep(5)
            except Exception as e:
                print("❌ Error:", e)

async def main_loop():
    while True:
        print("🔄 Проверка новостей...")
        await fetch_and_publish()
        await asyncio.sleep(1800)  # каждые 30 минут

if __name__ == "__main__":
    asyncio.run(main_loop())
