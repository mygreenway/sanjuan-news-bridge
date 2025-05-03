import os
import asyncio
import feedparser
import openai
from telegram import Bot
from telegram.constants import ParseMode

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CHANNEL_ID = "@sanjuan_online"
bot = Bot(token=BOT_TOKEN)

openai.api_key = OPENAI_API_KEY

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
    try:
        prompt = f"Resumen claro, completo y atractivo en español sobre esta noticia:\n\nTítulo: {title}\n\nContenido: {summary}"
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Eres un periodista español. Escribe publicaciones informativas y fáciles de entender."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print("❌ GPT error:", e)
        return summary  # fallback

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

            # Извлекаем изображение
            if "media_content" in entry:
                image_url = entry.media_content[0]["url"]
            elif "image" in entry:
                image_url = entry.image.get("href", "")

            improved_summary = await asyncio.to_thread(improve_summary_with_gpt, title, summary)

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
        print("🔄 Проверка новостей...")
        await fetch_and_publish()
        await asyncio.sleep(1800)  # каждые 30 минут

if __name__ == "__main__":
    asyncio.run(main_loop())
