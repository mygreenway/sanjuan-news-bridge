# -*- coding: utf-8 -*-
# Noticias España Bot — main.py (v1.7: 1 post per cycle, 60min interval, Madrid night sleep)

import os
import re
import io
import html
import time
import asyncio
import logging
import hashlib
import sqlite3
from datetime import datetime, timedelta
from urllib.parse import urlparse, urlunparse, urljoin

import feedparser
import requests
import trafilatura
from telegram import Bot, InputFile
from telegram.constants import ParseMode
from openai import OpenAI
from zoneinfo import ZoneInfo

# ===================== CONFIG =====================
CHANNEL = "@NoticiasEspanaHoy"
DB_PATH = "state.db"
HTTP_TIMEOUT = 15
CHECK_INTERVAL_MIN = 60                    # ← раз в 60 минут
USER_AGENT = "NoticiasEspanaBot/1.7 (+https://t.me/NoticiasEspanaHoy)"
OPENAI_MODEL = "gpt-4o-mini"

# Ночной «сон» по Мадриду
MADRID_TZ = ZoneInfo("Europe/Madrid")
QUIET_START_H = 0                          # с 00:00
QUIET_END_H = 8                           # до 07:00 (не включая)

# ===================== RSS SOURCES =====================
# Национальный минимум + акцент на Comunidad Valenciana и Alicante
RSS_FEEDS = [
    # 1) Национальный минимум
    "https://www.rtve.es/api/rss/portada",

    # 2) Региональные и локальные
    "https://www.20minutos.es/rss/comunidad-valenciana/",
    "https://www.informacion.es/rss/section/1056",
    "https://www.levante-emv.com/rss/section/13069",
    "https://www.lasprovincias.es/rss/2.0/portada",
    "https://www.alicante.es/es/noticias/rss",
]

# ===================== LOGGING =====================
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("NoticiasEspanaBot")

# ===================== DB =====================
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
              (url_h, title_h, url, title, int(datetime.now(MADRID_TZ).timestamp())))
    conn.commit()
    conn.close()

def cleanup_db(days: int = 7):
    cutoff = int((datetime.now(MADRID_TZ) - timedelta(days=days)).timestamp())
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM posts WHERE created_at < ?", (cutoff,))
    conn.commit()
    conn.close()

# ===================== HTTP & PARSING =====================
session = requests.Session()
session.headers.update({"User-Agent": USER_AGENT})

def http_get(url: str, extra_headers: dict | None = None) -> requests.Response:
    headers = session.headers.copy()
    if extra_headers:
        headers.update(extra_headers)
    return session.get(url, headers=headers, timeout=HTTP_TIMEOUT, allow_redirects=True)

def extract_main_image(html_text: str, base_url: str) -> str | None:
    # og: / twitter: / первая <img>
    for pattern in [
        r'<meta[^>]+property=["\']og:image["\'][^>]+content=["\']([^"\']+)["\']',
        r'<meta[^>]+name=["\']twitter:image["\'][^>]+content=["\']([^"\']+)["\']',
        r'<img[^>]+src=["\']([^"\']+)["\']',
    ]:
        m = re.search(pattern, html_text, re.IGNORECASE)
        if m:
            return urljoin(base_url, m.group(1))
    return None

def extract_text_and_image(url: str) -> tuple[str | None, str | None]:
    try:
        r = http_get(url)
        if r.status_code != 200 or not r.text:
            return None, None
        text = trafilatura.extract(r.text, include_comments=False, include_images=False, url=url)
        img = extract_main_image(r.text, url)
        return text, img
    except Exception as e:
        log.warning(f"extract_text_and_image error: {e}")
        return None, None

def download_image(url: str, referer: str | None = None) -> bytes | None:
    if not url:
        return None
    try:
        headers = {"Referer": referer} if referer else None
        r = http_get(url, headers)
        if r.status_code != 200:
            return None
        ctype = r.headers.get("Content-Type", "").lower()
        # Отправляем jpg/png (webp иногда не принимается как фото)
        if not any(x in ctype for x in ("image/jpeg", "image/jpg", "image/png")):
            return None
        if len(r.content) < 12000:  # отсечь совсем мелкие иконки
            return None
        return r.content
    except Exception:
        return None

# ===================== OPENAI =====================
def build_prompt(title_es: str, body_es: str) -> list[dict]:
    system = (
        "Ты переводчик-редактор. Переведи и сожми новость с испанского на русский. "
        "Факты без искажений; цифры/имена/даты сохранить; без домыслов. "
        "Сделай 2–3 абзаца по 1–2 предложения. Общая длина 450–550 символов. "
        "Первый абзац НЕ дублирует заголовок по смыслу. Нейтральный стиль. "
        "Выведи только текст тела без заголовка."
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

# ===================== LINK-IN-TEXT =====================
RU_STOPWORDS = {
    "это","как","так","его","ее","её","она","они","оно","когда","после","до","для",
    "чтобы","своих","свой","свои","может","будет","были","быть","при","где","который",
    "которые","также","этот","эта","эти","что","или","и","а","но","на","в","из","по",
    "под","над","от","до","без","у","про","же","ли","мы","вы","он","я","их","между"
}

def pick_anchor_word(title_ru: str, body_ru: str) -> str | None:
    first_para = body_ru.split("\n", 1)[0]
    pool = f"{title_ru} {first_para}"
    words = re.findall(r"[А-Яа-яЁёA-Za-z\-]{5,}", pool)
    for w in sorted(words, key=len, reverse=True):
        lw = w.lower().strip("-")
        if lw in RU_STOPWORDS:
            continue
        if re.search(rf"(?i)\b{re.escape(w)}\b", body_ru):
            return w
    return None

def inject_link_into_text(body_ru: str, source_url: str, anchor_word: str | None) -> str:
    if not anchor_word:
        return body_ru
    pattern = re.compile(rf"(?i)\b{re.escape(anchor_word)}\b")
    def repl(m):
        word = m.group(0)
        return f'<a href="{html.escape(source_url)}">{word}</a>'
    return pattern.sub(repl, body_ru, count=1)

# ===================== TEXT FORMAT =====================
def smart_trim(body: str, lo: int = 450, hi: int = 550) -> str:
    text = re.sub(r"\s+\n", "\n", body).strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    if len(text) <= hi:
        return text
    window = text[:hi]
    end = max(window.rfind("."), window.rfind("!"), window.rfind("?"))
    if end >= max(int(lo * 0.6), 200):
        return window[: end + 1].strip()
    return text[: hi - 1].rstrip() + "…"

def format_post(title_ru: str, body_ru: str, source_url: str) -> str:
    body_ru = smart_trim(body_ru, 450, 550)
    anchor = pick_anchor_word(title_ru, body_ru)
    body_linked = inject_link_into_text(body_ru, source_url, anchor)
    return (
        f"<b>{html.escape(title_ru.strip())}</b>\n\n"
        f"{body_linked}\n\n"
        f'<a href="https://t.me/NoticiasEspanaHoy">Новости Испания 🇪🇸</a>'
    ).strip()

def trim_caption(text: str, limit: int = 950) -> str:
    if len(text) <= limit:
        return text
    clean = re.sub(r"\s+", " ", text).strip()
    return clean[: limit - 1] + "…"

# ===================== TELEGRAM =====================
async def send_with_image(bot: Bot, text: str, image_bytes: bytes | None):
    if image_bytes:
        try:
            await bot.send_photo(
                chat_id=CHANNEL,
                photo=InputFile(io.BytesIO(image_bytes), filename="news.jpg"),
                caption=trim_caption(text, 950),
                parse_mode=ParseMode.HTML,
                disable_notification=True,
            )
            return True
        except Exception as e:
            log.warning(f"send_photo failed, fallback to text: {e}")
    try:
        await bot.send_message(
            chat_id=CHANNEL,
            text=text,
            parse_mode=ParseMode.HTML,
            disable_web_page_preview=True,  # без большого баннера
            disable_notification=True,
        )
        return True
    except Exception as e:
        log.error(f"send_message error: {e}")
        return False

# ===================== CORE: ONE POST PER CYCLE =====================
async def process_one_post(bot: Bot) -> bool:
    """
    Обходит ленты и публикует ПЕРВУЮ подходящую (ещё не публиковавшуюся) новость.
    Возвращает True, если что-то опубликовали.
    """
    for feed_url in RSS_FEEDS:
        try:
            feed = feedparser.parse(feed_url)
            for e in feed.entries:
                url = normalize_url(e.get("link") or e.get("id") or "")
                title = (e.get("title") or "").strip()
                if not url or not title:
                    continue
                if seen(url, title):
                    continue

                # тянем текст и картинку
                article_text, img_url = extract_text_and_image(url)
                if not article_text:
                    article_text = (e.get("summary") or e.get("description") or "").strip()
                if not article_text:
                    continue

                # перевод и сжатие
                title_ru, body_ru = translate_and_summarize(title, article_text)
                if not title_ru:
                    title_ru = title

                post_text = format_post(title_ru, body_ru, url)
                image_bytes = download_image(img_url, referer=url) if img_url else None

                await send_with_image(bot, post_text, image_bytes)
                mark_seen(url, title)
                log.info(f"Published: {title_ru}")
                return True  # ← публикуем только одну новость за этот цикл
        except Exception as ex:
            log.warning(f"Feed error {feed_url}: {ex}")
    return False

# ===================== NIGHT SLEEP =====================
def now_madrid() -> datetime:
    return datetime.now(MADRID_TZ)

def is_quiet_now() -> bool:
    h = now_madrid().hour
    # интервал [QUIET_START_H, QUIET_END_H)
    if QUIET_START_H <= QUIET_END_H:
        return QUIET_START_H <= h < QUIET_END_H
    return h >= QUIET_START_H or h < QUIET_END_H

def seconds_until_wakeup() -> int:
    now = now_madrid()
    today_wakeup = now.replace(hour=QUIET_END_H, minute=0, second=0, microsecond=0)
    if now < today_wakeup:
        target = today_wakeup
    else:
        target = (now + timedelta(days=1)).replace(hour=QUIET_END_H, minute=0, second=0, microsecond=0)
    return max(1, int((target - now).total_seconds()))

# ===================== SCHEDULER =====================
async def scheduler():
    tg_token = os.getenv("TELEGRAM_BOT_TOKEN")  # ← совместимо с твоими переменными
    if not tg_token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not set")
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set")

    bot = Bot(token=tg_token)
    ensure_db()
    cleanup_db(days=7)
    log.info("Noticias España Bot started")

    while True:
        if is_quiet_now():
            secs = seconds_until_wakeup()
            log.info(f"Quiet hours in Madrid. Sleeping for {secs//3600}h {secs%3600//60}m.")
            await asyncio.sleep(secs)
            continue

        try:
            posted = await process_one_post(bot)
            if not posted:
                log.info("No fresh posts this cycle.")
        except Exception as e:
            log.error(f"Cycle error: {e}")

        cleanup_db(days=7)
        await asyncio.sleep(CHECK_INTERVAL_MIN * 60)

def main():
    try:
        asyncio.run(scheduler())
    except KeyboardInterrupt:
        log.info("Stopped by user")

if __name__ == "__main__":
    main()
