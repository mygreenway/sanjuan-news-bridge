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

# Приоритет доменов (чем выше число — тем раньше обрабатываем фид)
DOMAIN_PRIORITY = {
    "elpais.com": 100,
    "rtve.es": 95,
    "elmundo.es": 92,
    "lavanguardia.com": 90,
    "abc.es": 88,
    "elconfidencial.com": 85,
    "20minutos.es": 80,
    "europapress.es": 78,
    "eldiario.es": 76,
    "publico.es": 70,
    "lasprovincias.es": 60,
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
FETCH_EVERY_SEC = 1800  # 30 minutes

CACHE_TITLES = "titles_cache.json"
CACHE_URLS = "urls_cache.json"
CACHE_FPS = "fps_cache.json"

EVENT_FPS_MAXLEN = 300
HAMMING_THRESHOLD_DUP = 4
HAMMING_THRESHOLD_MAYBE = 5

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

published_titles = load_set(CACHE_TITLES)
seen_urls = load_set(CACHE_URLS)
EVENT_FPS = load_fps(CACHE_FPS, EVENT_FPS_MAXLEN)

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
    # keep only <b>, <i>, <u>, <a href="">
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
    """Remove first sentence if it essentially repeats the title."""
    m = re.search(r'<b>(.*?)</b>', title_html, flags=re.S | re.I)
    title_plain = m.group(1) if m else ""
    def norm(x: str) -> str:
        return re.sub(r'\s+', ' ', re.sub(r'[^\wáéíóúñü ]+', '', (x or '').lower())).strip()
    t = norm(title_plain)
    if not t or not body_text:
        return body_text
    first = body_text.split('. ', 1)[0]
    if not first:
        return body_text
    f = norm(first)
    t_set, f_set = set(t.split()), set(f.split())
    jacc = len(t_set & f_set) / max(1, len(t_set | f_set))
    if jacc >= 0.6 or t.startswith(f) or f.startswith(t):
        return body_text[len(first):].lstrip('. ').lstrip()
    return body_text

def normalize_hashtags(s: str, limit: int = 3) -> str:
    """Возвращает до 3 нормализованных хештегов строчными, уникальными."""
    if not s:
        return ""
    raw = re.findall(r'#\w+|[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+', s)
    tags = []
    seen = set()
    for tok in raw:
        tag = tok.lower()
        if not tag.startswith('#'):
            tag = '#' + tag
        tag = re.sub(r'[^#\wáéíóúñü]', '', tag)
        if len(tag) < 2 or len(tag) > 32:
            continue
        if tag not in seen:
            seen.add(tag)
            tags.append(tag)
        if len(tags) >= limit:
            break
    return " ".join(tags)

# ---------------------- SIMHASH DEDUP --------------------------
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

# ---------------------- IMAGE EXTRACTION -----------------------
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

# ---------------------- ARTICLE FETCH -------------------------
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

# ---------------------- TOPIC/TAGS/EMOJI -----------------------
TOPIC_MAP = {
    "politica":      {"emoji": "🏛", "tags": "#españa #política"},
    "economia":      {"emoji": "💶", "tags": "#economía #negocios"},
    "deportes":      {"emoji": "⚽", "tags": "#deportes #españa"},
    "sucesos":       {"emoji": "🚨", "tags": "#sucesos #últimahora"},
    "ciencia":       {"emoji": "🔬", "tags": "#ciencia #investigación"},
    "cultura":       {"emoji": "🎭", "tags": "#cultura #arte"},
    "clima":         {"emoji": "🌦", "tags": "#clima #tiempo"},
    "internacional": {"emoji": "🌍", "tags": "#internacional"},
    "tecnologia":    {"emoji": "💻", "tags": "#tecnología #innovación"},
    "salud":         {"emoji": "🩺", "tags": "#salud #bienestar"},
}

COUNTRY_FLAG_MAP = {
    # Europa
    "españa":"🇪🇸","reino unido":"🇬🇧","uk":"🇬🇧","gran bretaña":"🇬🇧","francia":"🇫🇷","alemania":"🇩🇪",
    "italia":"🇮🇹","portugal":"🇵🇹","países bajos":"🇳🇱","holanda":"🇳🇱","bélgica":"🇧🇪","suiza":"🇨🇭",
    "austria":"🇦🇹","suecia":"🇸🇪","noruega":"🇳🇴","dinamarca":"🇩🇰","finlandia":"🇫🇮","irlanda":"🇮🇪",
    "polonia":"🇵🇱","grecia":"🇬🇷","chequia":"🇨🇿","hungría":"🇭🇺","rumanía":"🇷🇴","bulgaria":"🇧🇬",
    "serbia":"🇷🇸","croacia":"🇭🇷","eslovenia":"🇸🇮","eslovaquia":"🇸🇰","letonia":"🇱🇻","lituania":"🇱🇹",
    "estonia":"🇪🇪","ucrania":"🇺🇦","rusia":"🇷🇺","moldavia":"🇲🇩","georgia":"🇬🇪","armenia":"🇦🇲",
    "albania":"🇦🇱","bosnia":"🇧🇦","macedonia":"🇲🇰","montenegro":"🇲🇪",
    # América
    "estados unidos":"🇺🇸","eeuu":"🇺🇸","méxico":"🇲🇽","canadá":"🇨🇦","argentina":"🇦🇷","brasil":"🇧🇷",
    "chile":"🇨🇱","perú":"🇵🇪","colombia":"🇨🇴","uruguay":"🇺🇾","paraguay":"🇵🇾","ecuador":"🇪🇨",
    "bolivia":"🇧🇴","venezuela":"🇻🇪","panamá":"🇵🇦","cuba":"🇨🇺","república dominicana":"🇩🇴",
    "puerto rico":"🇵🇷",
    # Asia/África/ME
    "china":"🇨🇳","india":"🇮🇳","japón":"🇯🇵","corea del sur":"🇰🇷","corea del norte":"🇰🇵","turquía":"🇹🇷",
    "israel":"🇮🇱","palestina":"🇵🇸","arabia saudí":"🇸🇦","emiratos árabes unidos":"🇦🇪","qatar":"🇶🇦",
    "irán":"🇮🇷","iraq":"🇮🇶","siria":"🇸🇾","líbano":"🇱🇧","jordania":"🇯🇴","egipto":"🇪🇬",
    "marruecos":"🇲🇦","argelia":"🇩🇿","túnez":"🇹🇳","sudáfrica":"🇿🇦","nigeria":"🇳🇬","etiopía":"🇪🇹",
    "kenia":"🇰🇪",
    # Organizaciones
    "unión europea":"🇪🇺","ue":"🇪🇺"
}

def extract_countries_from_text(text: str) -> list[str]:
    t = (text or "").lower()
    t = re.sub(r'[^\wáéíóúñü ]+', ' ', t)
    found = []
    for name in COUNTRY_FLAG_MAP.keys():
        # грубый поиск по словам/фразам
        pattern = r'\b' + re.escape(name) + r'\b'
        if re.search(pattern, t):
            found.append(name)
    # уникальные, в порядке появления по длине (длинные фразы приоритетнее)
    uniq = []
    seen = set()
    for n in sorted(found, key=len, reverse=True):
        if n not in seen:
            seen.add(n)
            uniq.append(n)
    return list(reversed(uniq))  # более короткие ближе к концу

def choose_emoji_and_tags_by_topic(first_sentence: str, fallback_title: str) -> tuple[str, str, str]:
    """Возвращает (emoji, topic, topic_tags)"""
    # Тема из GPT-mini (дёшево и точно)
    async def detect_topic(text: str) -> str:
        prompt = (
            "Analiza la noticia y responde SOLO con una palabra de esta lista: "
            "politica, economia, deportes, sucesos, ciencia, cultura, clima, internacional, tecnologia, salud.\n"
            "Elige la que mejor describa el tema principal.\n"
            "Texto:\n" + (text or fallback_title)
        )
        resp = await openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=5
        )
        return (resp.choices[0].message.content or "").strip().lower()

    # Поскольку это sync оболочка, вернём корутину наружу
    return detect_topic(first_sentence or fallback_title)  # type: ignore

def final_emoji_for_topic(topic: str, first_sentence: str) -> str:
    base = TOPIC_MAP.get(topic, {"emoji": "📰"})["emoji"]
    if topic == "internacional":
        countries = extract_countries_from_text(first_sentence)
        if len(countries) == 1:
            return COUNTRY_FLAG_MAP.get(countries[0], base)
    return base

def merge_topic_and_gpt_tags(topic_tags: str, gpt_tags: str, limit: int = 3) -> str:
    # склейка и нормализация
    merged = " ".join([topic_tags or "", gpt_tags or ""]).strip()
    return normalize_hashtags(merged, limit=limit)

# --------------------- OPENAI HELPERS -------------------------
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

async def improve_summary_with_gpt(title: str, full_article: str, link: str) -> dict:
    lower_link = (link or "").lower()
    max_length = 2000 if any(w in lower_link for w in
        ["opinion", "opinión", "analisis", "análisis", "editorial", "tribuna"]) else 1500
    trimmed_article = full_article[:max_length]

    prompt = (
        "Escribe contenido para un post de Telegram sobre la noticia. Reglas ESTRICTAS:\n"
        "1) NO repitas el título en el texto: si el primer enunciado coincide con el título, reescríbelo.\n"
        "2) Texto: máx. 400 caracteres, claro y directo.\n"
        "3) No incluyas enlaces en el cuerpo.\n"
        "4) Devuelve JSON con dos campos: {\"body\": \"...\", \"tags\": \"#tag1 #tag2\"}\n"
        "5) No uses comillas dentro de los valores.\n\n"
        f"Título: {title}\n\nTexto fuente:\n{trimmed_article}"
    )

    resp = await openai_chat(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.6,
        max_tokens=420
    )
    raw = resp.choices[0].message.content.strip()
    import json as _json
    try:
        data = _json.loads(raw)
        body = str(data.get("body", "")).strip()
        tags = str(data.get("tags", "")).strip()
    except Exception:
        body = raw[:400]
        tags = ""
    return {"body": body, "tags": tags}

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
        temperature=0,
        max_tokens=10
    )
    ans = (resp.choices[0].message.content or "").strip().lower()
    return ans == "nueva"

# ------------------------- TELEGRAM ----------------------------
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
                caption = text[:1024]  # Telegram photo caption limit
                await bot.send_photo(channel, image_url, caption=caption, parse_mode=ParseMode.HTML)
            else:
                await bot.send_message(channel, text, parse_mode=ParseMode.HTML, disable_web_page_preview=False)
            return
        except Exception as e:
            logging.warning(f"Telegram send attempt {attempt+1} failed: {e}")
            await asyncio.sleep(delay)
            delay *= 2
    raise RuntimeError("Telegram send failed after retries")

# ---------------------- MAIN PIPELINE --------------------------
recent_summaries_for_gpt = deque(maxlen=50)

def first_paragraph(text: str) -> str:
    if not text:
        return ""
    para = text.strip().split("\n", 1)[0]
    return para[:400]

def feed_priority(url: str) -> int:
    try:
        host = urllib.parse.urlsplit(url).netloc
        parts = host.split('.')
        domain = ".".join(parts[-2:]) if len(parts) >= 2 else host
        return DOMAIN_PRIORITY.get(domain, 10)
    except Exception:
        return 10

async def fetch_and_publish():
    global published_titles, seen_urls, EVENT_FPS

    published_count = 0
    feeds_sorted = sorted(RSS_FEEDS, key=feed_priority, reverse=True)

    for feed_url in feeds_sorted:
        try:
            feed = feedparser.parse(feed_url)
        except Exception as e:
            logging.warning(f"feedparser error {feed_url}: {e}")
            continue

        for entry in feed.entries[:1]:
            if published_count >= MAX_PUBLICATIONS_PER_CYCLE:
                return

            raw_title = entry.title if hasattr(entry, "title") else ""
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

            fp = make_event_fingerprint(title, first_paragraph(full_article))
            if fp:
                is_dup = any(hamming(fp, old) <= HAMMING_THRESHOLD_DUP for old in EVENT_FPS)
                if is_dup:
                    continue
                maybe_dup = any(hamming(fp, old) == HAMMING_THRESHOLD_MAYBE for old in EVENT_FPS)
                if maybe_dup:
                    candidate_summary = (full_article[:600]).replace("\n", " ")
                    try:
                        still_new = await is_new_meaningful_gpt(candidate_summary, list(recent_summaries_for_gpt))
                        if not still_new:
                            continue
                    except Exception as e:
                        logging.warning(f"mini GPT dedupe failed, continue without it: {e}")

            # Текст поста от GPT (без ссылок/эмодзи)
            try:
                res = await improve_summary_with_gpt(title, full_article, clean_url)
            except Exception as e:
                logging.error(f"OpenAI improve_summary error: {e}")
                await notify_admin(f"❌ OpenAI error: {e}")
                continue

            title_html = f"<b>{safe_html_text(title)}</b>"
            body = safe_html_text(res["body"])
            body = drop_duplicate_title(title_html, body)
            gpt_tags_raw = res["tags"]

            # Определяем тему (gpt-4o-mini на 1-ю фразу) и подбираем эмодзи/теги
            first_sentence = (body.split('. ', 1)[0] or title)[:240]

            # detect_topic возвращает корутину -> ждём:
            topic = await choose_emoji_and_tags_by_topic(first_sentence, title)
            if not isinstance(topic, str) or topic not in TOPIC_MAP:
                topic = "internacional" if extract_countries_from_text(first_sentence) else "politica"

            emoji = final_emoji_for_topic(topic, first_sentence)
            topic_tags = TOPIC_MAP.get(topic, {"tags": "#noticias"}).get("tags", "#noticias")
            tags = merge_topic_and_gpt_tags(topic_tags, gpt_tags_raw, limit=3)

            leer_mas = f'🔗 <a href="{html.escape(clean_url)}">Leer más</a>'
            parts = [f"{emoji} {title_html}", "", body, "", leer_mas]
            if tags:
                parts.append(tags)
            final_text = "\n".join(p for p in parts if p is not None).strip()

            image_url = extract_image(entry)  # если пусто — публикуем текст (без заглушек)

            try:
                for channel in CHANNEL_IDS:
                    await send_with_retry(channel, image_url, final_text)

                seen_urls.add(clean_url)
                published_titles.add(norm_title)
                if fp:
                    EVENT_FPS.append(fp)
                save_set(CACHE_URLS, seen_urls)
                save_set(CACHE_TITLES, published_titles)
                save_fps(CACHE_FPS, EVENT_FPS)

                recent_summaries_for_gpt.append((full_article[:600]).replace("\n", " "))

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
