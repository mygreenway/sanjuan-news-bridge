import os
import asyncio
import feedparser
from telegram import Bot
from telegram.constants import ParseMode
from openai import AsyncOpenAI

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHANNEL_ID = "@sanjuan_online"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

bot = Bot(token=BOT_TOKEN)
openai = AsyncOpenAI(api_key=OPENAI_API_KEY)

# RSS-источники Испании
RSS_FEEDS = [
    "https://e00-elmundo.uecdn.es/elmundo/rss/portada.xml",
    "https://feeds.elpais.com/mrss-s/pages/ep/site/elpais.com/portada",
    "https://www.rtve.es/rss/portal/rss.xml",
    "https://www.20minutos.es/rss/",
    "https://www.europapress.es/rss/rss.aspx"
]

# Уникальные заголовки, чтобы не повторять публикации
published_titles = set()

# Функция для определения эмодзи и флага
def detect_emoji(text):
    text = text.lower()
    icon = "📰"
    flag = "🇪🇸"

    # Тематический эмодзи
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

    # Флаг по стране
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

# Расширяем краткое описание с помощью GPT
async def improve_summary_with_gpt(title, summary):
    prompt = (
        f"Mejora y amplía este resumen de noticia en español. No uses encabezados como 'Título' ni 'Resumen'. "
        f"Devuelve solo un texto claro, completo y atractivo para publicación en Telegram.\n\n"
        f"Título: {title}\n\nResumen: {summary}"
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

# Основной процесс
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

            emoji = detect_emoji(title + summary)
            improved_summary = await improve_summary_with_gpt(title, summary)

            hashtags = "#Noticias #España #SanJuan"
            text = (
                f"<b>{emoji} {title}</b>\n\n"
                f"{improved_summary}\n\n"
                f"👉 Haz clic <a href=\"{link}\">aquí para leer la noticia completa</a>\n\n"
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

# Цикл публикации каждые 30 минут
async def main_loop():
    while True:
        print("🔄 Comprobando noticias...")
        await fetch_and_publish()
        await asyncio.sleep(1800)

if __name__ == "__main__":
    asyncio.run(main_loop())
