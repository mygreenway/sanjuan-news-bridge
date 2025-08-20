# -*- coding: utf-8 -*-
import os
import re
import io
import sys
import json
import time
import html
import hashlib
import logging
import requests
import sqlite3
import asyncio
import feedparser
from urllib.parse import urljoin
from datetime import datetime
from collections import deque

from PIL import Image
from telegram import Bot
from telegram.constants import ParseMode
from openai import AsyncOpenAI

# ===================== CONFIG =====================
CHANNEL = "@NoticiasEspanaHoy"
CHECK_INTERVAL_MIN = 60   # публикация каждые 60 минут
DB_PATH = "state.db"
HTTP_TIMEOUT = 15
USER_AGENT = "NoticiasEspanaBot/1.4 (+https://t.me/NoticiasEspanaHoy)"
OPENAI_MODEL = "gpt-4o-mini"

# ===================== RSS SOURCES =====================
RSS_FEEDS = [
    # Национальный
    "https://www.rtve.es/api/rss/portada",
    # Региональные и локальные
    "https://www.20minutos.es/rss/comunidad-valenciana/",
    "https://www.informacion.es/rss/section/1056",
    "https://www.levante-emv.com/rss/section/13069",
    "https://www.lasprovincias.es/rss/2.0/portada",
    "https://www.alicante.es/es/noticias/rss",
]

# ===================== LOGGING =====================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger(__name__)

# ===================== DB =====================
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("CREATE TABLE IF NOT EXISTS posted (uid TEXT PRIMARY KEY)")
    conn.commit()
    conn.close()

def already_posted(uid: str) -> bool:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM posted WHERE uid=?", (uid,))
    row = cur.fetchone()
    conn.close()
    return row is not None

def mark_posted(uid: str):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("INSERT OR IGNORE INTO posted(uid) VALUES (?)", (uid,))
    conn.commit()
    conn.close()

# ===================== HTTP =====================
def http_get(url: str, headers: dict = None) -> requests.Response:
    headers = headers or {}
    if "User-Agent" not in headers:
        headers["User-Agent"] = USER_AGENT
    return requests.get(url, headers=headers, timeout=HTTP_TIMEOUT)

# ===================== IMAGES =====================
def _pick_from_srcset(srcset: str, base_url: str) -> str | None:
    try:
        items = []
        for part in srcset.split(","):
            seg = part.strip().split()
            if not seg:
                continue
            url_part = seg[0]
            w = 0
            if len(seg) > 1 and seg[1].endswith("w"):
                try:
                    w = int(seg[1][:-1])
                except Exception:
                    w = 0
            items.append((w, urljoin(base_url, url_part)))
        if not items:
            return None
        items.sort(key=lambda x: x[0], reverse=True)
        return items[0][1]
    except Exception:
        return None

def extract_main_image(html_text: str, base_url: str) -> str | None:
    for pattern in [
        r'<meta[^>]+property=["\']og:image:secure_url["\'][^>]+content=["\']([^"\']+)["\']',
        r'<meta[^>]+property=["\']og:image["\'][^>]+content=["\']([^"\']+)["\']',
        r'<meta[^>]+name=["\']twitter:image["\'][^>]+content=["\']([^"\']+)["\']',
    ]:
        m = re.search(pattern, html_text, re.IGNORECASE)
        if m:
            return urljoin(base_url, m.group(1))

    m = re.search(r'<img[^>]+srcset=["\']([^"\']+)["\'][^>]*>', html_text, re.IGNORECASE)
    if m:
        candidate = _pick_from_srcset(m.group(1), base_url)
        if candidate:
            return candidate

    m = re.search(r'<img[^>]+src=["\']([^"\']+)["\']', html_text, re.IGNORECASE)
    if m:
        return urljoin(base_url, m.group(1))

    return None

def image_from_rss(entry) -> str | None:
    for key in ("media_content", "media_thumbnail"):
        val = entry.get(key)
        if isinstance(val, list):
            for item in val:
                if isinstance(item, dict) and item.get("url"):
                    return item["url"]
    for l in entry.get("links", []):
        if isinstance(l, dict) and l.get("rel") == "enclosure" and l.get("href"):
            return l["href"]
    return None

def download_image(url: str, referer: str | None = None) -> bytes | None:
    if not url:
        return None
    try:
        headers = {"Accept": "image/*,*/*"}
        if referer:
            headers["Referer"] = referer
        r = http_get(url, headers)
        if r.status_code != 200:
            return None
        ctype = r.headers.get("Content-Type", "").lower()
        raw = r.content
        if len(raw) < 12000:
            return None
        if "image/jpeg" in ctype or "image/jpg" in ctype:
            return raw
        if any(fmt in ctype for fmt in ("image/png", "image/webp")):
            try:
                im = Image.open(io.BytesIO(raw)).convert("RGB")
                buf = io.BytesIO()
                im.save(buf, format="JPEG", quality=88, optimize=True)
                return buf.getvalue()
            except Exception:
                return None
        try:
            im = Image.open(io.BytesIO(raw)).convert("RGB")
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=88, optimize=True)
            return buf.getvalue()
        except Exception:
            return None
    except Exception:
        return None

# ===================== GPT =====================
openai_client = AsyncOpenAI()

async def summarize(title: str, text: str) -> tuple[str, str]:
    prompt = f"""
Ты — новостной редактор. Сократи и переведи новость с испанского на русский для Telegram-канала.
Формат:
<b>{title}</b>
1–2 абзаца текста, максимум 600 символов. Суть и удобочитаемость.
Вставь ссылку <a href='{text[:200]}'>источник</a> в ключевое слово внутри текста.
В конце — короткая подпись:

Новости Испания 🇪🇸
"""
    resp = await openai_client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
        max_tokens=500
    )
    out = resp.choices[0].message.content.strip()
    parts = out.split("\n", 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return title, out

# ===================== TELEGRAM =====================
bot = Bot(os.environ["TELEGRAM_BOT_TOKEN"])

async def send_post(title: str, body: str, link: str, img_url: str | None):
    caption = f"{body.strip()}\n\nНовости Испания 🇪🇸"
    img_data = download_image(img_url, link) if img_url else None
    if img_data:
        await bot.send_photo(CHANNEL, photo=img_data, caption=caption, parse_mode=ParseMode.HTML)
    else:
        await bot.send_message(CHANNEL, text=caption, parse_mode=ParseMode.HTML)

# ===================== PROCESS =====================
async def process_one_post():
    for feed_url in RSS_FEEDS:
        try:
            d = feedparser.parse(feed_url)
            for e in d.entries:
                uid = hashlib.sha256(e.link.encode()).hexdigest()
                if already_posted(uid):
                    continue

                url = e.link
                img_url = image_from_rss(e)

                try:
                    r = http_get(url)
                    if r.status_code == 200:
                        html_text = r.text
                        if not img_url:
                            img_url = extract_main_image(html_text, url)
                except Exception:
                    pass

                title_ru, body_ru = await summarize(e.title, url)
                await send_post(title_ru, body_ru, url, img_url)
                mark_posted(uid)
                return  # публикуем только одну новость за цикл
        except Exception as ex:
            log.error(f"Ошибка при обработке {feed_url}: {ex}")

# ===================== MAIN LOOP =====================
async def main_loop():
    init_db()
    while True:
        await process_one_post()
        await asyncio.sleep(CHECK_INTERVAL_MIN * 60)

if __name__ == "__main__":
    asyncio.run(main_loop())
