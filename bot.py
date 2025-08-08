import os
import re
import json
import time
import html
import asyncio
import logging
import hashlib
import urllib.parse
from collections import deque

import feedparser
import trafilatura
from telegram import Bot
from telegram.constants import ParseMode
from openai import AsyncOpenAI

# ---------------------------- ЛОГИ ----------------------------
logging.basicConfig(
    filename='bot_log.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# ------------------------- НАСТРОЙКИ --------------------------
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ADMIN_CHAT_ID = os.getenv("ADMIN_CHAT_ID")

CHANNEL_IDS = ["@NoticiasEspanaHoy"]

# RSS-источники
RSS_FEEDS = [
    "https://e00-elmundo.uecdn.es/elmundo/rss/portada.xml",
    "https://feeds.elpais.com/mrss/s/pages/ep/site/elpais.com/portada",
    "https://www.rtve.es/rss/portal/rss.xml",
    "https://www.20minutos.es/rss/",
    "https://www.europapress.es/rss/rss.aspx",
    "https://www.abc.es/rss/feeds/abcPortada.xml",
    "https://www.lavanguardia.com/mvc/feed/rss/home",
    "https://www.elconfidencial.com/rss/espana.xml",
    "https://www.eldiario.es/rss/",
    "https://www.publico.es/rss/",
    "https://www.lasprovincias.es/rss/2.0/portada/index.rss",
]

# Ограничения публикаций
MAX_PUBLICATIONS_PER_CYCLE = 5
SLEEP_BETWEEN_POSTS_SEC = 5
FETCH_EVERY_SEC = 1800  # 30 минут

# Кэш-файлы
CACHE_TITLES = "titles_cache.json"       # set[str] нормализованные заголовки
CACHE_URLS = "urls_cache.json"           # set[str] нормализованные URL
CACHE_FPS = "fps_cache.json"             # list[int] последние simhash (ограничены по длине)

# Параметры дедупликации
EVENT_FPS_MAXLEN = 300      # хранить до 300 отпечатков событий
HAMMING_THRESHOLD_DUP = 4   # <=4 считаем дубль
HAMMING_THRESHOLD_MAYBE = 5 # опциональный GPT-проверочный коридор (редко используется)

# ----------------------- ИНИЦИАЛИЗАЦИЯ ------------------------
if not BOT_TOKEN or not OPENAI_API_KEY:
    raise RuntimeError("Missing TELEGRAM_BOT_TOKEN or OPENAI_API_KEY")

bot = Bot(token=BOT_TOKEN)
openai = AsyncOpenAI(api_key=OPENAI_API_KEY)

# ---------------------- УТИЛИТЫ/КЭШИ --------------------------
def load_set(path: str) -> set:
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return set(json.load(f))
        except Exception as e:
            logging.warning(f"Cache load warning ({path}): {e}")
    return set()

def save_set(path: str, data: set):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(sorted(list(data))[-10000:], f, ensure_ascii=False, indent=2)
    except Exception as e:
        logging.error(f"Cache save error ({path}): {e}")

def load_fps(path: str, maxlen: int) -> deque:
    dq = deque(maxlen=maxlen)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                items = json.load(f)
                for it in items[-maxlen:]:
                    if isinstance(it, int):
                        dq.append(it)
        except Exception as e:
            logging.warning(f"FPS cache load warning: {e}")
    return dq

def save_fps(path: str, dq: deque):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(list(dq)[-EVENT_FPS_MAXLEN:], f, ensure_ascii=False, indent=2)
    except Exception as e:
        logging.error(f"FPS cache save error: {e}")

published_titles = load_set(CACHE_TITLES)
seen_urls = load_set(CACHE_URLS)
EVENT_FPS = load_fps(CACHE_FPS, EVENT_FPS_MAXLEN)

# ---------------------- ТЕКСТ/HTML УТИЛИТЫ --------------------
def normalize_title(title: str) -> str:
    t = re.sub(r'\s+', ' ', (title or '').strip().lower())
    t = re.sub(r'[^\wáéíóúñü ]+', '', t)
    return t

def normalize_url(url: str) -> str:
    if not url:
        return ""
    u = urllib.parse.urlsplit(url)
    qs = urllib.parse.parse_qs(u.query)
    qs = {k: v for k, v in qs.items() if not k.lower().startswith(('utm_', 'fbclid', 'gclid', 'mc_eid'))}
    query = urllib.parse.urlencode(qs, doseq=True)
    clean = urllib.parse.urlunsplit((u.scheme, u.netloc, u.path.rstrip('/'), query, ''))
    return clean.lower()

def safe_html_text(s: str) -> str:
    """Экранируем HTML, оставляя только <b>, <i>, <u>, <a href="">."""
    # временные маркеры
    s = s.replace('<b>', '§B§').replace('</b>', '§/B§')
    s = s.replace('<i>', '§I§').replace('</i>', '§/I§')
    s = s.replace('<u>', '§U§').replace('</u>', '§/U§')
    s = re.sub(r'<a\s+href="([^"]+)">', r'§A§\1§', s)
    s = s.replace('</a>', '§/A§')
    # escape
    s = html.escape(s)
    # восстановить разрешённые теги
    s = s.replace('§B§', '<b>').replace('§/B§', '</b>')
    s = s.replace('§I§', '<i>').replace('§/I§', '</i>')
    s = s.replace('§U§', '<u>').replace('§/U§', '</u>')
    s = re.sub(r'§A§([^§]+)§', r'<a href="\1">', s)
    s = s.replace('§/A§', '</a>')
    return s

def inject_link(improved_text: str, url: str) -> str:
    """Пытаемся вставить ссылку во 2-е слово 2-й строки, иначе добавляем отдельной строкой."""
    lines = [l for l in (improved_text or "").split("\n")]
    link = f'<a href="{html.escape(url)}">más</a>'
    if len(lines) >= 2 and lines[1].strip():
        words = lines[1].split()
        if len(words) >= 2:
            # безопасно обернём второе слово ссылкой
            words[1] = f'<a href="{html.escape(url)}">{safe_html_text(words[1])}</a>'
            lines[1] = " ".join(words)
            return "\n".join(lines)
        else:
            lines.append(link)
            return "\n".join(lines)
    else:
        lines.append(link)
        return "\n".join(lines)

# --------------------- SIMHASH ДЕДУПЛИКАЦИЯ -------------------
SPANISH_STOP = set("""
de la que el en y a los del se las por un para con no una su al lo como más pero sus le ya o este
sí porque esta entre cuando muy sin sobre también me hasta hay donde quien desde todo nos durante
todos uno les ni contra otros ese eso ante ellos e esto mí antes algunos qué unos yo otro otras otra
él tanto esa estos mucho quienes nada muchos cual poco ella estar estas algunas algo nosotros mi mis
tú te ti tu tus ellas nosotras vosotras vosotros os mío mía míos mías tuyo tuya tuyos tuyas suyo suya
suyos suyas nuestro nuestra nuestros nuestras vuestro vuestra vuestros vuestras esos esas estoy estás
está estamos estáis están esté estés estemos estéis estén estaré estarás estará estaremos estaréis
estarán estaba estabas estaba estábamos estabais estaban estuve estuviste estuvo estuvimos estuvisteis
estuvieron estuviera estuvieras estuviéramos estuvierais estuvieran estuviese estuvieses
estuviésemos estuvieseis estuviesen estando estado estada estados estadas estad
""".split())

def tokenize_core(text: str) -> list:
    text = (text or "").lower()
    text = re.sub(r'https?://\S+', ' ', text)
    text = re.sub(r'[^0-9a-záéíóúñü ]+', ' ', text)
    toks = [w for w in text.split() if w not in SPANISH_STOP and (len(w) >= 4 or w.isdigit())]
    return toks

def simhash64(tokens: list) -> int:
    v = [0] * 64
    for t in tokens:
        h = int(hashlib.md5(t.encode('utf-8')).hexdigest(), 16)
        for i in range(64):
            v[i] += 1 if (h >> i) & 1 else -1
    out = 0
    for i, score in enumerate(v):
        if score > 0:
            out |= (1 << i)
    return out

def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()

def make_event_fingerprint(title: str, first_para: str) -> int:
    toks = tokenize_core(title) + tokenize_core(first_para)
    if len(toks) > 40:
        toks = toks[:40]
    return simhash64(toks) if toks else 0

# -------------------- ИЗВЛЕЧЕНИЕ КАРТИНОК ---------------------
def extract_image(entry) -> str:
    # 1) media_content
    try:
        mc = entry.get("media_content", [])
        if mc and mc[0].get("url"):
            return mc[0]["url"]
    except Exception:
        pass
    # 2) media_thumbnail
    try:
        mt = entry.get("media_thumbnail", [])
        if mt and mt[0].get("url"):
            return mt[0]["url"]
    except Exception:
        pass
    # 3) links из фида
    for l in entry.get("links", []):
        if l.get("type", "").startswith("image/") and l.get("href"):
            return l["href"]
    # 4) из summary HTML
    for field in ("summary", "summary_detail"):
        html_blob = entry.get(field, "") or ""
        m = re.search(r'<img[^>]+src=["\']([^"\']+)["\']', str(html_blob), re.I)
        if m:
            return m.group(1)
    return ""

# ------------------- ЗАГРУЗКА ПОЛНОГО ТЕКСТА ------------------
def get_full_article(url: str, retries: int = 2) -> str:
    for attempt in range(1 + retries):
        try:
            downloaded = trafilatura.fetch_url(url, no_ssl=True)
            if not downloaded:
                time.sleep(0.5)
                continue
            text = trafilatura.extract(
                downloaded,
                include_comments=False,
                include_tables=False,
                favor_precision=True,
            )
            if text and len(text.split()) >= 60:
                return text
        except Exception as e:
            logging.warning(f"trafilatura fail: {e}")
        time.sleep(0.7)
    return ""

# -------------------- OPENAI ХЕЛПЕРЫ/РЕТРАИ -------------------
async def openai_chat(messages, model="gpt-4o", temperature=0.6, max_tokens=400, retries=2):
    delay = 1.0
    for attempt in range(retries + 1):
        try:
            resp = await openai.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return resp
        except Exception as e:
            logging.warning(f"OpenAI error attempt {attempt+1}: {e}")
            await asyncio.sleep(delay)
            delay *= 2
    raise RuntimeError("OpenAI chat failed after retries")

async def improve_summary_with_gpt(title: str, full_article: str, link: str) -> str:
    lower_link = (link or "").lower()
    if any(w in lower_link for w in ["opinion", "opinión", "analisis", "análisis", "editorial", "tribuna"]):
        max_length = 2000
    else:
        max_length = 1500

    trimmed_article = full_article[:max_length]
    prompt = (
        "Escribe una publicación para Telegram sobre la siguiente noticia. Sigue este formato ESTRICTAMENTE:\n\n"
        "1) Primera línea: un emoji temático y el título en negrita usando <b> ... </b> (NO uses **).\n"
        "2) Luego un párrafo (máx. 400 caracteres) que resuma claramente la noticia. No repitas el título.\n"
        "3) No añadas ningún enlace. Solo texto.\n"
        "4) Última línea: 2 o 3 hashtags relevantes, separados por espacios.\n\n"
        f"Título: {title}\n\nTexto:\n{trimmed_article}"
    )

    resp = await openai_chat(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.6,
        max_tokens=400
    )
    text = resp.choices[0].message.content.strip()
    # Безопасный размер для caption фото
    return text[:1000]

async def is_new_meaningful_gpt(candidate_summary: str, recent_summaries: list[str]) -> bool:
    """Опциональная дешёвая проверка через gpt-4o-mini, если расстояние пограничное."""
    joined = "\n".join(f"- {s}" for s in recent_summaries[-10:])  # сравнение только с 10 последними
    prompt = (
        "Analiza si la siguiente noticia ya fue publicada. "
        "Considera 'repetida' si trata sobre el mismo evento, aunque cambien palabras, títulos o cifras.\n\n"
        f"Últimas publicadas:\n{joined}\n\nNueva:\n{candidate_summary}\n\n"
        "Responde solo con 'nueva' o 'repetida'."
    )
    resp = await openai_chat(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=10
    )
    ans = (resp.choices[0].message.content or "").strip().lower()
    return ans == "nueva"

# ------------------------- ТЕЛЕГРАМ ---------------------------
async def notify_admin(message: str):
    if not ADMIN_CHAT_ID:
        return
    try:
        await bot.send_message(chat_id=ADMIN_CHAT_ID, text=message)
    except Exception as e:
        logging.error(f"Error notifying admin: {e}")

async def send_with_retry(channel: str, image_url: str, text: str):
    delay = 1.0
    for attempt in range(3):
        try:
            if image_url:
                # caption у фото ≤ 1024
                caption = text[:1024]
                await bot.send_photo(channel, image_url, caption=caption, parse_mode=ParseMode.HTML)
            else:
                # без картинки оставляем превью ссылок включённым (красивые карточки)
                await bot.send_message(channel, text, parse_mode=ParseMode.HTML, disable_web_page_preview=False)
            return
        except Exception as e:
            logging.warning(f"Telegram send attempt {attempt+1} failed: {e}")
            await asyncio.sleep(delay)
            delay *= 2
    raise RuntimeError("Telegram send failed after retries")

# --------------------- ОСНОВНАЯ ЛОГИКА ------------------------
recent_summaries_for_gpt = deque(maxlen=50)  # короткие резюме для редкой GPT-проверки

def first_paragraph(text: str) -> str:
    if not text:
        return ""
    para = text.strip().split("\n", 1)[0]
    return para[:400]

async def fetch_and_publish():
    global published_titles, seen_urls, EVENT_FPS

    published_count = 0

    for feed_url in RSS_FEEDS:
        try:
            feed = feedparser.parse(feed_url)
        except Exception as e:
            logging.warning(f"feedparser error {feed_url}: {e}")
            continue

        # берём только свежий топ: [:1] — экономим запросы/спам
        for entry in feed.entries[:1]:
            if published_count >= MAX_PUBLICATIONS_PER_CYCLE:
                return

            raw_title = entry.title if hasattr(entry, "title") else ""
            # Некоторые фиды пихают источник перед заголовком: "El País: Título..."
            title = re.sub(r'^[^:|]+[|:]\s*', '', raw_title).strip()

            norm_title = normalize_title(title)
            clean_url = normalize_url(getattr(entry, "link", ""))

            # Фильтр 0: точные дубли по URL/заголовку
            if not clean_url:
                continue
            if clean_url in seen_urls or norm_title in published_titles:
                continue

            # Полный текст
            full_article = get_full_article(clean_url)
            # если не получилось — fallback на summary
            if not full_article:
                full_article = getattr(entry, "summary", "") or ""
            # минимальная длина контента
            if len(full_article.split()) < 80:
                continue

            # Дешёвый смысловой фильтр: simhash title + 1-й абзац
            fp = make_event_fingerprint(title, first_paragraph(full_article))
            if fp:
                # проверка на схожесть с последними отпечатками
                is_dup = any(hamming(fp, old) <= HAMMING_THRESHOLD_DUP for old in EVENT_FPS)
                if is_dup:
                    continue
                # редкая пограничная проверка через GPT-mini
                maybe_dup = any(hamming(fp, old) == HAMMING_THRESHOLD_MAYBE for old in EVENT_FPS)
                if maybe_dup:
                    # дешёвое короткое резюме для сравнения
                    candidate_summary = (full_article[:600]).replace("\n", " ")
                    try:
                        still_new = await is_new_meaningful_gpt(candidate_summary, list(recent_summaries_for_gpt))
                        if not still_new:
                            continue
                    except Exception as e:
                        logging.warning(f"mini GPT dedupe failed, continue without it: {e}")

            # Генерация финального текста
            try:
                improved_text = await improve_summary_with_gpt(title, full_article, clean_url)
            except Exception as e:
                logging.error(f"OpenAI improve_summary error: {e}")
                await notify_admin(f"❌ OpenAI error: {e}")
                continue

            # Безопасный HTML и вставка ссылки
            improved_text = safe_html_text(improved_text)
            improved_text = inject_link(improved_text, clean_url)

            # Подпись-бренд
            improved_text += '\n\n<a href="https://t.me/NoticiasEspanaHoy">📡 Noticias de España</a>'

            # Картинка
            image_url = extract_image(entry)

            # Публикация
            try:
                for channel in CHANNEL_IDS:
                    await send_with_retry(channel, image_url, improved_text)
                # Обновляем кэши только после успеха
                seen_urls.add(clean_url)
                published_titles.add(norm_title)
                if fp:
                    EVENT_FPS.append(fp)
                save_set(CACHE_URLS, seen_urls)
                save_set(CACHE_TITLES, published_titles)
                save_fps(CACHE_FPS, EVENT_FPS)

                # Сохраним короткий конспект для редкой GPT-проверки (дёшево)
                recent_summaries_for_gpt.append((full_article[:600]).replace("\n", " "))

                published_count += 1
                await asyncio.sleep(SLEEP_BETWEEN_POSTS_SEC)

            except Exception as e:
                logging.error(f"Telegram error: {e}")
                await notify_admin(f"❌ Error publicación: {e}")
                # не записываем в кэши если не получилось отправить

async def main_loop():
    while True:
        logging.info("🔄 Comprobando noticias...")
        try:
            await fetch_and_publish()
        except Exception as e:
            logging.error(f"fetch_and_publish crashed: {e}")
            await notify_admin(f"❌ Ciclo falló: {e}")
        await asyncio.sleep(FETCH_EVERY_SEC)

if __name__ == "__main__":
    asyncio.run(main_loop())
