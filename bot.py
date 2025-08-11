import os
import re
import json
import time
import html
import asyncio
import logging
import hashlib
import urllib.parse
import difflib
from collections import deque

import feedparser
import trafilatura
from telegram import Bot
from telegram.constants import ParseMode
from openai import AsyncOpenAI

# ---------------------------- LOGGING ----------------------------
logging.basicConfig(
    filename='bot_log.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# --------------------------- SETTINGS ----------------------------
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ADMIN_CHAT_ID = os.getenv("ADMIN_CHAT_ID")

CHANNEL_IDS = ["@NoticiasEspanaHoy"]
CHANNEL_SIGNATURE = '<a href="https://t.me/NoticiasEspanaHoy">📡 Noticias de España</a>'

# Приоритет доменов (выше — раньше)
DOMAIN_PRIORITY = {
    "elpais.com": 100, "rtve.es": 95, "elmundo.es": 92, "lavanguardia.com": 90,
    "abc.es": 88, "elconfidencial.com": 85, "20minutos.es": 80, "europapress.es": 78,
    "eldiario.es": 76, "publico.es": 70, "lasprovincias.es": 60,
}

RSS_FEEDS = [
    "https://feeds.elpais.com/mrss/s/pages/ep/site/elpais.com/portada",
    "https://www.rtve.es/rss/portal/rss.xml",
    "https://e00-elmundo.uecdn.es/elmundo/rss/portada.xml",
    "https://www.lavanguardia.com/mvc/feed/rss/home",
    "https://www.abc.es/rss/feeds/abcPortada.xml",
    "https://www.elconfidencial.com/rss/espana.xml",
    "https://www.20minutos.es/rss/",
    "https://www.europapress.es/rss/rss.aspx",
    "https://www.eldiario.es/rss/",
    "https://www.publico.es/rss/",
    "https://www.lasprovincias.es/rss/2.0/portada/index.rss",
]

MAX_PUBLICATIONS_PER_CYCLE = 5
SLEEP_BETWEEN_POSTS_SEC = 5
FETCH_EVERY_SEC = 1800  # 30 минут

CACHE_TITLES = "titles_cache.json"
CACHE_URLS = "urls_cache.json"
CACHE_FPS = "fps_cache.json"
CACHE_EVENT_KEYS = "event_keys.json"  # кэш канонических ключей события

EVENT_FPS_MAXLEN = 300
HAMMING_THRESHOLD_DUP = 6
HAMMING_THRESHOLD_MAYBE = 8

# ------------------------- INIT CLIENTS --------------------------
if not BOT_TOKEN or not OPENAI_API_KEY:
    raise RuntimeError("Missing TELEGRAM_BOT_TOKEN or OPENAI_API_KEY")

bot = Bot(token=BOT_TOKEN)
openai = AsyncOpenAI(api_key=OPENAI_API_KEY)

# --------------------------- CACHES ------------------------------
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
        logging.error(f"FPS cache save error ({path}): {e}")

def load_list(path: str) -> list:
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_list(path: str, data: list, maxlen: int = 400):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(list(data)[-maxlen:], f, ensure_ascii=False, indent=2)
    except Exception as e:
        logging.error(f"Cache save error ({path}): {e}")

published_titles = load_set(CACHE_TITLES)
seen_urls = load_set(CACHE_URLS)
EVENT_FPS = load_fps(CACHE_FPS, EVENT_FPS_MAXLEN)
EVENT_KEYS = deque(load_list(CACHE_EVENT_KEYS), maxlen=400)

# ----------------------- TEXT/HTML UTILS ------------------------
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
    s = s.replace('<b>', '§B§').replace('</b>', '§/B§')
    s = s.replace('<i>', '§I§').replace('</i>', '§/I§')
    s = s.replace('<u>', '§U§').replace('</u>', '§/U§')
    s = re.sub(r'<a\s+href="([^"]+)">', r'§A§\1§', s)
    s = s.replace('</a>', '§/A§')
    s = html.escape(s)
    s = s.replace('§B§', '<b>').replace('§/B§', '</b>')
    s = s.replace('§I§', '<i>').replace('§/I§', '</i>')
    s = s.replace('§U§', '<u>').replace('§/U§', '</u>')
    s = re.sub(r'§A§([^§]+)§', r'<a href="\1">', s)
    s = s.replace('§/A§', '</a>')
    return s

def drop_duplicate_title(title_html: str, body_text: str) -> str:
    m = re.search(r'<b>(.*?)</b>', title_html, flags=re.S | re.I)
    title_plain = m.group(1) if m else ""
    def norm(x: str) -> str:
        return re.sub(r'\s+', ' ', re.sub(r'[^\wáéíóúñü ]+', '', (x or '').lower())).strip()
    t = norm(title_plain)
    if not t or not body_text:
        return body_text
    body = re.sub(r"[“”«»]", '"', body_text)
    body = re.sub(r"\s+\.\s+", ". ", body)
    first = body.split('. ', 1)[0]
    if not first:
        return body
    f = norm(first)
    t_set, f_set = set(t.split()), set(f.split())
    jacc = len(t_set & f_set) / max(1, len(t_set | f_set))
    if jacc >= 0.6 or t.startswith(f) or f.startswith(t):
        return body[len(first):].lstrip('. ').lstrip()
    return body

# ---------------------- SIMHASH + JACCARD ----------------------
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

SPANISH_STOP_MIN = SPANISH_STOP | {
    "gobierno","plan","ciudad","seguridad","ministro","presidente","nacional","oficial","medida",
    "grupo","región","local","nueva","nuevo","según","contra","tras","donde","mientras","entre"
}

def tokenize_core(text: str) -> list:
    text = (text or "").lower()
    text = re.sub(r'https?://\S+', ' ', text)
    text = re.sub(r'[^0-9a-záéíóúñü ]+', ' ', text)
    return [w for w in text.split() if w not in SPANISH_STOP and (len(w) >= 4 or w.isdigit())]

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

RECENT_BODIES = deque(maxlen=120)

def normalize_tokens_for_jaccard(text: str) -> set[str]:
    t = (text or "").lower()
    t = re.sub(r'https?://\S+', ' ', t)
    t = re.sub(r'[^0-9a-záéíóúñü ]+', ' ', t)
    toks = [w for w in t.split() if (len(w) >= 4 or w.isdigit()) and w not in SPANISH_STOP_MIN]
    return set(toks[:120])

def is_jaccard_dup(new_body: str, threshold: float = 0.55) -> bool:
    cand = normalize_tokens_for_jaccard(new_body)
    if not cand:
        return False
    for prev in RECENT_BODIES:
        inter = len(cand & prev)
        union = len(cand | prev) or 1
        if inter / union >= threshold:
            return True
    return False

# ---------------------- ARTICLE FETCH -------------------------
def extract_image(entry) -> str:
    try:
        mc = entry.get("media_content", [])
        if mc and mc[0].get("url"):
            return mc[0]["url"]
    except Exception:
        pass
    try:
        mt = entry.get("media_thumbnail", [])
        if mt and mt[0].get("url"):
            return mt[0]["url"]
    except Exception:
        pass
    for l in entry.get("links", []):
        if l.get("type", "").startswith("image/") and l.get("href"):
            return l["href"]
    for field in ("summary", "summary_detail"):
        html_blob = entry.get(field, "") or ""
        m = re.search(r'<img[^>]+src=["\']([^"\']+)["\']', str(html_blob), re.I)
        if m:
            return m.group(1)
    return ""

def get_full_article(url: str, retries: int = 2) -> str:
    for attempt in range(1 + retries):
        try:
            downloaded = trafilatura.fetch_url(url, no_ssl=True)
            if not downloaded:
                time.sleep(0.5); continue
            text = trafilatura.extract(
                downloaded, include_comments=False, include_tables=False, favor_precision=True
            )
            if text and len(text.split()) >= 60:
                return text
        except Exception as e:
            logging.warning(f"trafilatura fail: {e}")
        time.sleep(0.7)
    return ""

def first_paragraph(text: str) -> str:
    if not text:
        return ""
    para = text.strip().split("\n", 1)[0]
    return para[:400]

# --------------------- OPENAI HELPERS -------------------------
async def openai_chat(messages, model="gpt-4o-mini", temperature=0.3, max_tokens=500, retries=2):
    delay = 1.0
    for attempt in range(retries + 1):
        try:
            resp = await openai.chat.completions.create(
                model=model, messages=messages, temperature=temperature, max_tokens=max_tokens
            )
            return resp
        except Exception as e:
            logging.warning(f"OpenAI error attempt {attempt+1}: {e}")
            await asyncio.sleep(delay); delay *= 2
    raise RuntimeError("OpenAI chat failed after retries")

def extract_json_obj(raw: str) -> str | None:
    cleaned = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", raw.strip(), flags=re.S)
    m = re.search(r"\{.*\}", cleaned, flags=re.S)
    return m.group(0) if m else None

def normalize_hashtags(s: str, limit: int = 3) -> str:
    if not s:
        return ""
    raw = re.findall(r'#\w+|[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+', s)
    tags, seen = [], set()
    for tok in raw:
        tag = tok.lower()
        if not tag.startswith('#'):
            tag = '#' + tag
        tag = re.sub(r'[^#\wáéíóúñü]', '', tag)
        if 2 <= len(tag) <= 32 and tag not in seen:
            seen.add(tag); tags.append(tag)
        if len(tags) >= limit:
            break
    return " ".join(tags)

async def generate_full_post_with_gpt(source_title: str, full_article: str) -> dict:
    """
    Возвращает JSON: {"title": "...", "emoji": "…", "body": "...", "tags": "#a #b"}
    Правила:
    - Заголовок: испанский, 70–110 символов, информативный, без кавычек и ЭМОДЗИ, первая буква заглавная.
    - Эмодзи: ровно 1 (флаг допускается, если явно одна главная страна; иначе тематическое).
    - Текст: <= 400 символов, без ссылок, без повторения смысла заголовка в первом предложении.
    - Теги: 1–3 штуки максимум. Можно словами без # — мы нормализуем.
    - Ответ строго JSON без ```.
    """
    max_length = 1800
    trimmed_article = (full_article or "")[:max_length]

    prompt = (
        "Escribe los campos para un post de Telegram sobre esta noticia. Reglas ESTRICTAS:\n"
        "1) Titular en español, 70–110 caracteres, informativo y específico, sin comillas ni emojis. Primera letra en mayúscula.\n"
        "2) Un solo emoji temático (puede ser bandera SOLO si hay un único país protagonista claro; si hay varios o no está claro, usa temático, no banderas).\n"
        "3) Cuerpo conciso (máximo 400 caracteres), sin enlaces. No repitas el título en el primer enunciado: si coincide, reformúlalo.\n"
        "4) 1–3 etiquetas temáticas.\n"
        "5) Devuelve JSON en texto plano, sin bloques de código ni ```.\n"
        "   Formato: {\"title\":\"...\",\"emoji\":\"...\",\"body\":\"...\",\"tags\":\"#tag1 #tag2\"}\n\n"
        f"Título origen: {source_title}\n\n"
        f"Texto fuente:\n{trimmed_article}"
    )

    resp = await openai_chat(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3, max_tokens=520
    )
    raw = (resp.choices[0].message.content or "").strip()
    jtxt = extract_json_obj(raw) or raw

    # Блок страховки парсинга и формата
    title, emoji, body, tags = source_title, "📰", "", ""
    try:
        data = json.loads(jtxt)
        title = re.sub(r'[\"“”«»]+', '', str(data.get("title", "")).strip())
        emoji = str(data.get("emoji", "")).strip()
        body = str(data.get("body", "")).strip()
        tags = str(data.get("tags", "")).strip()
    except Exception as e:
        logging.warning(f"JSON parse fallback: {e}; raw={raw[:200]}")

    # Мини-валидации
    if not title or len(title) < 40:
        fp = first_paragraph(full_article)
        title = (fp[:100] + "…") if fp else (source_title[:100] or "Actualidad")

    # Оставляем один символ эмодзи (некоторые флаги занимают 2 code units — это ок для Telegram)
    # Если GPT вернул несколько — обрежем до первого "графемного" символа на глаз (~2 юникодных)
    emoji = emoji[:2] if emoji else "📰"
    tags = normalize_hashtags(tags, limit=3)

    # Тело без повторения заголовка ещё раз проверим ниже (drop_duplicate_title)
    return {"title": title, "emoji": emoji, "body": body, "tags": tags}

async def is_new_meaningful_gpt(candidate_summary: str, recent_summaries: list[str]) -> bool:
    joined = "\n".join(f"- {s}" for s in recent_summaries[-10:])
    prompt = (
        "Analiza si la siguiente noticia ya fue publicada. "
        "Considera 'repetida' si trata sobre el mismo evento, aunque cambien palabras, títulos o cifras.\n\n"
        f"Últimas publicadas:\n{joined}\n\nNueva:\n{candidate_summary}\n\n"
        "Responde solo con 'nueva' o 'repetida'."
    )
    resp = await openai_chat(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0, max_tokens=10
    )
    ans = (resp.choices[0].message.content or "").strip().lower()
    return ans == "nueva"

# ---------------------- EVENT KEY (жёсткий дедуп) ----------------------
async def make_event_key(title: str, first_paragraph_text: str) -> str:
    base = (title + " " + first_paragraph_text)[:600]
    prompt = (
        "Genera un ID canónico (slug) para esta noticia. Reglas:\n"
        "- Solo minusculas, a-z, 0-9 y guiones.\n"
        "- 4–8 tokens clave (actor, accion, objeto, lugar/fecha si aporta).\n"
        "- Sin nombres de medios, sin comillas, sin acentos.\n"
        "- El MISMO evento contado en distintos medios debe dar el MISMO slug.\n"
        f"Texto:\n{base}\n\nDevuelve SOLO el slug."
    )
    try:
        resp = await openai_chat(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0, max_tokens=20
        )
        slug = (resp.choices[0].message.content or "").strip().lower()
        slug = re.sub(r"[^a-z0-9\-]+", "-", slug)
        slug = re.sub(r"-{2,}", "-", slug).strip("-")
    except Exception:
        txt = re.sub(r"[^a-z0-9 ]+", " ", (title + " " + first_paragraph_text).lower())
        toks = [t for t in txt.split() if len(t) >= 4][:8]
        slug = "-".join(toks) or hashlib.md5(txt.encode()).hexdigest()[:16]
    return slug[:80]

def is_event_key_dup(new_key: str, keys: deque, ratio: float = 0.86) -> bool:
    for k in keys:
        if new_key == k:
            return True
        if difflib.SequenceMatcher(None, new_key, k).ratio() >= ratio:
            return True
    return False

# ------------------------- TELEGRAM ----------------------------
async def notify_admin(message: str):
    if not ADMIN_CHAT_ID:
        return
    try:
        await bot.send_message(chat_id=ADMIN_CHAT_ID, text=message)
    except Exception as e:
        logging.error(f"Error notifying admin: {e}")

async def send_message_or_photo(channel: str, image_url: str | None, caption_or_text: str):
    delay = 1.0
    for attempt in range(3):
        try:
            if image_url:
                await bot.send_photo(channel, image_url, caption=caption_or_text, parse_mode=ParseMode.HTML)
            else:
                await bot.send_message(channel, caption_or_text, parse_mode=ParseMode.HTML, disable_web_page_preview=False)
            return
        except Exception as e:
            logging.warning(f"Telegram send attempt {attempt+1} failed: {e}")
            await asyncio.sleep(delay); delay *= 2
    raise RuntimeError("Telegram send failed after retries")

# ---------------------- MAIN PIPELINE --------------------------
recent_summaries_for_gpt = deque(maxlen=50)

def feed_priority(url: str) -> int:
    try:
        host = urllib.parse.urlsplit(url).netloc
        parts = host.split('.')
        domain = ".".join(parts[-2:]) if len(parts) >= 2 else host
        return DOMAIN_PRIORITY.get(domain, 10)
    except Exception:
        return 10

async def fetch_and_publish():
    global published_titles, seen_urls, EVENT_FPS, EVENT_KEYS

    published_count = 0
    feeds_sorted = sorted(RSS_FEEDS, key=feed_priority, reverse=True)

    for feed_url in feeds_sorted:
        try:
            feed = feedparser.parse(feed_url)
        except Exception as e:
            logging.warning(f"feedparser error {feed_url}: {e}")
            continue

        # по одному самому свежему элементу из каждого фида (как и раньше)
        for entry in feed.entries[:1]:
            if published_count >= MAX_PUBLICATIONS_PER_CYCLE:
                return

            raw_title = entry.title if hasattr(entry, "title") else ""
            # срез префикса типа "Sección: Título"
            title = re.sub(r'^[^:|]+[|:]\s*', '', raw_title).strip()

            norm_title = normalize_title(title)
            clean_url = normalize_url(getattr(entry, "link", ""))

            if not clean_url:
                continue
            if clean_url in seen_urls or norm_title in published_titles:
                continue

            full_article = get_full_article(clean_url)
            if not full_article:
                full_article = getattr(entry, "summary", "") or ""
            if len(full_article.split()) < 80:
                continue

            # === Жёсткий дедуп по каноническому ключу события ===
            fp_first_para = first_paragraph(full_article)
            event_key = await make_event_key(title, fp_first_para)
            if is_event_key_dup(event_key, EVENT_KEYS):
                continue

            # === simhash дедуп по событию ===
            fp = make_event_fingerprint(title, fp_first_para)
            if fp:
                is_dup = any(hamming(fp, old) <= HAMMING_THRESHOLD_DUP for old in EVENT_FPS)
                if is_dup:
                    continue
                maybe_dup = any(hamming(fp, old) <= HAMMING_THRESHOLD_MAYBE for old in EVENT_FPS)
                if maybe_dup:
                    candidate_summary = (full_article[:600]).replace("\n", " ")
                    try:
                        still_new = await is_new_meaningful_gpt(candidate_summary, list(recent_summaries_for_gpt))
                        if not still_new:
                            continue
                    except Exception as e:
                        logging.warning(f"mini GPT dedupe failed, continue without it: {e}")

            # === ЕДИНЫЙ GPT-ВЫЗОВ: заголовок + эмодзи + тело + теги
            try:
                g = await generate_full_post_with_gpt(title, full_article)
                gpt_title = g["title"]
                gpt_emoji = g["emoji"]
                body = g["body"]
                tags = g["tags"]
            except Exception as e:
                logging.error(f"OpenAI post generation error: {e}")
                await notify_admin(f"❌ OpenAI error: {e}")
                continue

            title_html = f"<b>{safe_html_text(gpt_title)}</b>"
            body = safe_html_text(body)
            body = drop_duplicate_title(title_html, body)

            # дешёвый Jaccard по телу поста
            if is_jaccard_dup(body):
                continue

            # СБОРКА ХВОСТА: «Leer más» отдельной строкой, затем теги, затем подпись канала
            leer_mas = f'<a href="{html.escape(clean_url)}">Leer más</a>'
            tail_parts = [leer_mas]
            if tags:
                tail_parts.append(tags.lower())
            tail_parts.append(CHANNEL_SIGNATURE)
            tail = "\n\n".join(tail_parts)

            image_url = extract_image(entry)
            head = f"{gpt_emoji} {title_html}\n\n"

            if image_url:
                budget = 1024 - len(head) - len("\n\n") - len(tail)
                trimmed_body = body[:max(0, budget)]
                payload = head + trimmed_body + "\n\n" + tail
            else:
                payload = head + body + "\n\n" + tail

            try:
                for channel in CHANNEL_IDS:
                    await send_message_or_photo(channel, image_url, payload)

                # --- после успешной публикации фиксируем всё в кэшах
                seen_urls.add(clean_url)
                published_titles.add(normalize_title(gpt_title) or norm_title)
                if fp:
                    EVENT_FPS.append(fp)
                EVENT_KEYS.append(event_key)

                save_set(CACHE_URLS, seen_urls)
                save_set(CACHE_TITLES, published_titles)
                save_fps(CACHE_FPS, EVENT_FPS)
                save_list(CACHE_EVENT_KEYS, list(EVENT_KEYS))

                recent_summaries_for_gpt.append((full_article[:600]).replace("\n", " "))
                RECENT_BODIES.append(normalize_tokens_for_jaccard(body))

                published_count += 1
                await asyncio.sleep(SLEEP_BETWEEN_POSTS_SEC)

            except Exception as e:
                logging.error(f"Telegram error: {e}")
                await notify_admin(f"❌ Error publicación: {e}")

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
