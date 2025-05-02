import os
import asyncio
from telegram import Bot
from telegram.constants import ParseMode

bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
channel_id = "@sanjuan_online"

bot = Bot(token=bot_token)

async def main():
    await bot.send_message(
        chat_id=channel_id,
        text="✅ *Тестовая публикация:* San Juan Bot успешно запущен!",
        parse_mode=ParseMode.MARKDOWN
    )

if __name__ == "__main__":
    asyncio.run(main())
