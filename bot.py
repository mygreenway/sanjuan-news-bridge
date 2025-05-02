import os
from telegram import Bot
from telegram.constants import ParseMode

bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
channel_id = "@sanjuan_online"

bot = Bot(token=bot_token)

# Тестовое сообщение
bot.send_message(
    chat_id=channel_id,
    text="✅ *Тестовая публикация:* San Juan Bot успешно запущен!",
    parse_mode=ParseMode.MARKDOWN
)
