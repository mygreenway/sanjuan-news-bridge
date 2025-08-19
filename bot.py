# -*- coding: utf-8 -*-
# Noticias España Bot — main.py (async fix for PTB v20)

import os
import re
import html
import time
import asyncio
import logging
import hashlib
import sqlite3
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse, urlunparse

import feedparser
import requests
import trafilatura
from telegram import Bot
from telegram.constants import ParseMode

from openai import OpenAI

# ---------------------------- CONFIG ----------------------------
CHANNEL = "@NoticiasEspanaHoy"
CHECK_INTERVAL_MIN = 30
DB_PATH = "state.db"
HTTP_TIMEOUT = 15
USER_AGENT = "NoticiasEspanaBot/1.0 (+https://t.me/NoticiasEspanaHoy)"
OPENAI_MODEL = "gpt-4o-mini"

RSS_FEEDS = [
    "https://elpais.com/rss/elpais/portada.xml",
    "https://e00-elmundo.uecdn.es/elmundo/rss/portada.xml",
    "https://www.20minutos.es/rss/",
    "https://www.abc.es/rss/feeds/abc_ultima.xml",
    "https://www.rtve.es/api/rss/portada",
    "https://www.informacion.es/rss/section/1056",
    "https://www.levante-emv.com/rss/section/13069",
    "https://www.lasprovincias.es/rss/2.0/portada",
]

# ---------------------------- LOGGING ----------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("noticias-espana")

# ---------------------------- DB ----------------------------
def sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", "ignore")).hexdigest()

def normalize_url(u: str) -> str:
    try:
        p = urlparse(u)
        clean_query = "&".join(sorted([q for q in p.query.split("&") if q and not q.lower().startswith(("utm_", "fbclid"))]))
        return urlunparse(p._replace(query=clean_query))
    except Exception:
        return u

def ensure_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS posts(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url_hash TEXT UNIQUE,
            title_hash TEXT,
            source_url TEXT,
            title TEXT,
            created_at INTEGER
        )
    """)
    c.execute("CREATE INDEX IF NOT EXISTS idx_created_at ON posts(created_at)")
    conn.commit()
    conn.close()

def seen(url: str, title: str) -> bool:
    url_h = sha256(normalize_url(url))
    title_h = sha256(title.strip().lower())
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT 1 FROM posts WHERE url_hash=? OR title_hash=?", (url_h, title_h))
    row = c.fetchone()
    conn.close()
    return row is not None

def mark_seen(url: str, title: str):
    url_h = sha256(normalize_url(url))
    title_h = sha256(title.strip().lower())
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT OR IGNORE INTO posts(url_hash, title_hash, source_url, title, created_at) VALUES (?,?,?,?,?)",
              (url_h, title_h, url, title, int(time.time())))
    conn.commit()
    conn.close()

def cleanup_db(days: int = 7):
    cutoff = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp())
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM posts WHERE created_at < ?", (cutoff,))
    conn.commit()
    conn.close()

# ---------------------------- HTTP ----------------------------
session = requests.Session()
session.headers.update({"User-Agent": USER_AGENT})

def fetch_url(url: str) -> requests.Response:
    return session.get(url, timeout=HTTP_TIMEOUT, allow_redirects=True)

def extract_main_image(html_text: str) -> str | None:
    og = re.search(r'<meta[^>]+property=["\']og:image["\'][^>]+content=["\']([^"\']+)["\']', html_text, re.IGNORECASE)
    if og:
        return og.group(1)
    m = re.search(r'<img[^>]+src=["\']([^"\']+)["\']', html_text, re.IGNORECASE)
    return m.group(1) if m else None

def extract_text(url: str) -> tuple[str | None, str | None]:
    try:
        r = fetch_url(url)
        if r.status_code != 200 or not r.text:
            return None, None
        text = trafilatura.extract(r.text, include_comments=False, include_images=False, url=url)
        img = extract_main_image(r.text)
        return text, img
    except Exception as e:
        log.warning(f"extract_text error: {e}")
        return None, None

# ---------------------------- OPENAI ----------------------------
def build_prompt(title_es: str, body_es: str) -> list[dict]:
    system = (
        "Ты переводчик-редактор. Переведи и сожми новость с испанского на русский. "
        "Факты без искажений; цифры/имена/даты сохранить; без домыслов. "
        "2–4 абзаца, первый НЕ дублирует заголовок по смыслу. Нейтральный стиль. "
        "Выводи только текст тела без заголовка."
    )
    user = f"Заголовок (ES): {title_es}\n\nСтатья (ES):\n{body_es[:8000]}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]

def translate_and_summarize(title_es: str, body_es: str) -> tuple[str, str]:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Перевод заголовка
    try:
        t = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "Переведи заголовок с испанского на русский. Выведи только перевод."},
                {"role": "user", "content": title_es.strip()[:300]},
            ],
            temperature=0.2,
        )
        title_ru = t.choices[0].message.content.strip()
    except Exception as e:
        log.warning(f"Title translation error: {e}")
        title_ru = title_es

    # Резюме тела
    try:
        m = client.chat.completions.create(
            model=OPENAI_MODEL, messages=build_prompt(title_es, body_es), temperature=0.2
        )
        body_ru = m.choices[0].message.content.strip()
    except Exception as e:
        log.warning(f"Body summarize error: {e}")
        body_ru = ""

    return title_ru, body_ru

# ---------------------------- FORMAT ----------------------------
def format_post(title_ru: str, body_ru: str, source_url: str) -> str:
    parts = [
        f"<b>{html.escape(title_ru.strip())}</b>\n",
        body_ru.strip() if body_ru else "",
        f'\n\n<a href="{html.escape(source_url)}">Читать источник</a>',
        '\n\n<a href="https://t.me/NoticiasEspanaHoy">Новости Испания 🇪🇸</a>',
    ]
    return "\n".join([p for p in parts if p]).strip()

def trim_for_caption(text: str, limit: int = 1024) -> str:
    if len(text) <= limit:
        return text
    clean = re.sub(r"\s+", " ", text).strip()
    if len(clean) <= limit:
        return clean
    return clean[: limit - 1] + "…"

# ---------------------------- PUBLISH (ASYNC) ----------------------------
async def post_to_channel(bot: Bot, text: str, image_url: str | None):
    try:
        if image_url:
            caption = trim_for_caption(text, 1024)
            if len(caption) == len(text):
                await bot.send_photo(
                    chat_id=CHANNEL,
                    photo=image_url,
                    caption=caption,
                    parse_mode=ParseMode.HTML,
                    disable_notification=True,
                )
                return
        await bot.send_message(
            chat_id=CHANNEL,
            text=text,
            parse_mode=ParseMode.HTML,
            disable_web_page_preview=False,
            disable_notification=True,
        )
    except Exception as e:
        log.error(f"Telegram send error: {e}")

# ---------------------------- PIPELINE (ASYNC) ----------------------------
async def process_entry(bot: Bot, entry):
    url = normalize_url(entry.get("link") or entry.get("id") or "")
    title = (entry.get("title") or "").strip()
    if not url or not title:
        return
    if seen(url, title):
        return

    article_text, image_url = extract_text(url)
    if not article_text:
        article_text = (entry.get("summary") or entry.get("description") or "").strip()
    if not article_text:
        return

    title_ru, body_ru = translate_and_summarize(title, article_text)
    if not title_ru:
        title_ru = title

    post_text = format_post(title_ru, body_ru, url)
    await post_to_channel(bot, post_text, image_url)

    mark_seen(url, title)
    log.info(f"Published: {title_ru}")

async def check_feeds_once(bot: Bot):
    for feed_url in RSS_FEEDS:
        try:
            f = feedparser.parse(feed_url)
            if not f.entries:
                continue
            for e in list(f.entries)[-10:]:
                await process_entry(bot, e)
        except Exception as e:
            log.warning(f"Feed error {feed_url}: {e}")

# ---------------------------- MAIN LOOP ----------------------------
async def scheduler():
    tg_token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not tg_token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not set")
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set")

    bot = Bot(token=tg_token)

    ensure_db()
    cleanup_db(days=7)

    log.info("Noticias España Bot started")

    await check_feeds_once(bot)

    while True:
        await asyncio.sleep(CHECK_INTERVAL_MIN * 60)
        await check_feeds_once(bot)
        cleanup_db(days=7)

def main():
    try:
        asyncio.run(scheduler())
    except KeyboardInterrupt:
        log.info("Stopped by user")

if __name__ == "__main__":
    main()
