# -*- coding: utf-8 -*-
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

# Приоритет доменов
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
CACHE_EVENT_KEYS = "event_keys.json"
CACHE_EVENT_CLUSTER = "event_cluster.json"   # slug -> {ts,url,title}
CACHE_LAST_TITLES = "last_titles.json"       # сырые последние заголовки

EVENT_FPS_MAXLEN = 300
HAMMING_THRESHOLD_DUP = 6
HAMMING_THRESHOLD_MAYBE = 8
EVENT_CLUSTER_TTL_SEC = 48 * 3600  # 48 часов

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
        logging.error(f"Cache save error ({path}): {e}")

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

def load_dict(path: str) -> dict:
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logging.warning(f"Cache load warning ({path}): {e}")
    return {}

def save_dict(path: str, data: dict, max_items: int = 2000):
    try:
        items = sorted(data.items(), key=lambda kv: kv[1].get("ts", 0))[-max_items:]
        to_save = {k: v for k, v in items}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(to_save, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logging.error(f"Cache save error ({path}): {e}")

published_titles = load_set(CACHE_TITLES)
seen_urls = load_set(CACHE_URLS)
EVENT_FPS = load_fps(CACHE_FPS, EVENT_FPS_MAXLEN)
EVENT_KEYS = deque(load_list(CACHE_EVENT_KEYS), maxlen=1000)
EVENT_CLUSTER = load_dict(CACHE_EVENT_CLUSTER)
last_titles_raw = load_list(CACHE_LAST_TITLES)

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

def mask_link_in_body(body_text: str, url: str) -> str:
    """Спрятать ссылку в первом подходящем слове тела поста."""
    if not body_text or not url:
        return body_text
    plain = re.sub(r'<[^>]+>', '', body_text)
    words = plain.split()
    anchor_idx = -1
    for i, w in enumerate(words):
        ww = re.sub(r'[^0-9a-záéíóúñü]', '', w.lower())
        if ww and (len(ww) >= 6 or ww.isdigit()) and ww not in SPANISH_STOP_MIN:
            anchor_idx = i
            break
    if anchor_idx == -1:
        first, *rest = body_text.split('. ', 1)
        linked = first + f' (<a href="{html.escape(url)}">detalles</a>)'
        return (linked + ('. ' + rest[0] if rest else '')).strip()

    def replacer(match):
        token = match.group(0)
        idx = replacer.i
        replacer.i += 1
        if idx == anchor_idx:
            return f'<a href="{html.escape(url)}">{html.escape(token)}</a>'
        return token
    replacer.i = 0
    return re.sub(r'([^\W_]+)', replacer, body_text, flags=re.UNICODE)

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

# ---------------------- ARTICLE FETCH (HTML+текст) --------------
def get_full_article(url: str, retries: int = 2) -> tuple[str, str]:
    last_html = ""
    for attempt in range(1 + retries):
        try:
            downloaded = trafilatura.fetch_url(url, no_ssl=True)
            if downloaded:
                last_html = downloaded
                text = trafilatura.extract(
                    downloaded, include_comments=False, include_tables=False, favor_precision=True
                )
                if text and len(text.split()) >= 60:
                    return text, downloaded
        except Exception as e:
            logging.warning(f"trafilatura fail: {e}")
        time.sleep(0.7)
    return "", last_html

def first_paragraph(text: str) -> str:
    if not text:
        return ""
    para = text.strip().split("\n", 1)[0]
    return para[:400]

# --------------------- EMOJI ENGINE (мульти-флаги) --------------
TOPIC_EMOJI = {
    "politica": "🏛", "economia_up": "📈", "economia_down": "📉", "economia": "💶",
    "deportes_futbol": "⚽", "deportes_basket": "🏀", "deportes_tenis": "🎾",
    "sucesos": "🚨", "incendio": "🔥", "accidente_trafico": "🚗",
    "ciencia": "🔬", "tecnologia": "🤖", "salud": "🩺", "clima_lluvia": "🌧️",
    "clima_calor": "🔥", "clima_viento": "🌬️", "clima_general": "🌦️",
    "internacional": "🌍", "default": "📰"
}

SPORT_PATTERNS = [
    (re.compile(r'\b(\d+)\s*[-:]\s*(\d+)\b'), "deportes_futbol"),
]
SPORT_KEYWORDS = {
    "deportes_futbol": ["liga","champions","copa del rey","fichaje","gol","penalti","real madrid","barça","fc barcelona","atlético","selección"],
    "deportes_basket": ["acb","euroliga","baloncesto","nba","triple","rebote"],
    "deportes_tenis": ["tenis","wimbledon","roland garros","us open","australian open","match point","sets"]
}
CLIMA = {
    "clima_lluvia": ["lluvia","tormenta","granizo","borrasca","inundación"],
    "clima_calор": ["ola de calor","temperaturas récord","temperatura máxima","bochorno","alerta roja por calor"],
    "clima_viento": ["viento","rachas","temporal","ciclón","huracán"],
}
ECON_UP = ["sube","crece","récord","máximo","acelera","alza"]
ECON_DOWN = ["cae","baja","desplome","mínimo","recesión","en contracción"]
SUCE = {
    "incendio": ["incendio","fuego","forestal"],
    "accidente_trafico": ["accidente de tráfico","choque","colisión","atropello"],
    "sucesos": ["detenido","agresión","tiroteo","linchamiento","investigación policial","allanamiento"]
}
TEC = ["tecnología","ia","inteligencia artificial","ciberataque","app","plataforma","software","algoritmo","robot"]
SALUD = ["salud","hospital","sanidad","vacuna","brotes","virus","gripe","covid"]

COUNTRY_FLAG_MAP = {
    "españa":"🇪🇸","reino unido":"🇬🇧","uk":"🇬🇧","gran bretaña":"🇬🇧","francia":"🇫🇷","alemania":"🇩🇪",
    "italia":"🇮🇹","portugal":"🇵🇹","países bajos":"🇳🇱","holanda":"🇳🇱","bélgica":"🇧🇪",
    "suiza":"🇨🇭","austria":"🇦🇹","suecia":"🇸🇪","noruega":"🇳🇴","dinamarca":"🇩🇰","finlandia":"🇫🇮",
    "irlanda":"🇮🇪","polonia":"🇵🇱","grecia":"🇬🇷","chequia":"🇨🇿","hungría":"🇭🇺","rumanía":"🇷🇴",
    "bulgaria":"🇧🇬","serbia":"🇷🇸","croacia":"🇭🇷","eslovenia":"🇸🇮","eslovaquia":"🇸🇰",
    "letonia":"🇱🇻","lituania":"🇱🇹","estonia":"🇪🇪","ucrania":"🇺🇦","рusia":"🇷🇺","moldavia":"🇲🇩",
    "georgia":"🇬🇪","armenia":"🇦🇲","albania":"🇦🇱","bosnia":"🇧🇦","macedonia":"🇲🇰","montenegro":"🇲🇪",
    "estados unidos":"🇺🇸","eeuu":"🇺🇸","méxico":"🇲🇽","canadá":"🇨🇦","argentina":"🇦🇷","brasil":"🇧🇷",
    "chile":"🇨🇱","perú":"🇵🇪","colombia":"🇨🇴","uruguay":"🇺🇾","paraguay":"🇵🇾","ecuador":"🇪🇨",
    "bolivia":"🇧🇴","venezuela":"🇻🇪","panamá":"🇵🇦","china":"🇨🇳","india":"🇮🇳","japón":"🇯🇵",
    "corea del sur":"🇰🇷","turquía":"🇹🇷","israel":"🇮🇱","palestина":"🇵🇸","emiratos árabes unidos":"🇦🇪",
    "qatar":"🇶🇦","irán":"🇮🇷","iraq":"🇮🇶","egipto":"🇪🇬","marruecos":"🇲🇦","sudáfrica":"🇿🇦"
}

MAX_FLAGS = 4

def _has_any(text: str, keywords: list[str]) -> bool:
    t = text.lower()
    return any(k in t for k in keywords)

def extract_countries_from_text(text: str) -> list[str]:
    t = (text or "").lower()
    t = re.sub(r'[^\wáéíóúñü ]+', ' ', t)
    found = [name for name in COUNTRY_FLAG_MAP if re.search(r'\b'+re.escape(name)+r'\b', t)]
    uniq, seen = [], set()
    for n in sorted(found, key=len, reverse=True):
        if n not in seen:
            seen.add(n); uniq.append(n)
    return list(reversed(uniq))

def reorder_countries(countries: list[str]) -> list[str]:
    if "españa" in countries:
        return ["españa"] + [c for c in countries if c != "españa"]
    return countries

def flags_for_countries(countries: list[str]) -> str:
    if not countries:
        return ""
    cc = reorder_countries(countries)
    if len(cc) == 1:
        return COUNTRY_FLAG_MAP.get(cc[0], "")
    if 2 <= len(cc) <= MAX_FLAGS:
        out, seen = [], set()
        for c in cc:
            f = COUNTRY_FLAG_MAP.get(c)
            if f and f not in seen:
                out.append(f); seen.add(f)
        return "".join(out) if out else ""
    return "🌍"

def _economy_signal(text: str) -> str | None:
    t = text.lower()
    has_num = bool(re.search(r'(\b\d{1,3}(?:[.,]\d{1,2})?\s*%|\b\d{1,3}(?:[.,]\d{3})+|\b\d+[.,]?\d*\s*(?:€|euros|millones|miles))', t))
    econ_word = _has_any(t, ["inflación","ipc","paro","desempleo","pib","salario","hacienda","déficit","deuda","mercado","bolса","ibex"])
    if not (has_num or econ_word):
        return None
    if any(w in t for w in ECON_UP):
        return "economia_up"
    if any(w in t for w in ECON_DOWN):
        return "economia_down"
    return "economia"

def final_emoji_deterministic(text: str) -> str:
    t = (text or "").lower()
    for rx, key in SPORT_PATTERNS:
        if rx.search(t):
            return TOPIC_EMOJI[key]
    for key, kws in SPORT_KEYWORDS.items():
        if _has_any(t, kws):
            return TOPIC_EMOJI[key]
    for key, kws in CLIMA.items():
        if _has_any(t, kws):
            return TOPIC_EMOJI.get(key, TOPIC_EMOJI["clima_general"])
    econ = _economy_signal(t)
    if econ:
        fl = flags_for_countries(extract_countries_from_text(t))
        return fl or TOPIC_EMOJI[econ]
    for key, kws in SUCE.items():
        if _has_any(t, kws):
            fl = flags_for_countries(extract_countries_from_text(t))
            return fl or TOPIC_EMOJI[key]
    if _has_any(t, TEC):
        fl = flags_for_countries(extract_countries_from_text(t))
        return fl or TOPIC_EMOJI["tecnologia"]
    if _has_any(t, SALUD):
        fl = flags_for_countries(extract_countries_from_text(t))
        return fl or TOPIC_EMOJI["salud"]
    if _has_any(t, ["gobierno","congreso","senado","ministro","presidente","decreto","ley","elecciones","coalición","tribunal"]):
        fl = flags_for_countries(extract_countries_from_text(t))
        return fl or TOPIC_EMOJI["politica"]
    fl = flags_for_countries(extract_countries_from_text(t))
    if fl:
        return fl
    return TOPIC_EMOJI["default"]

# --------------------- OPENAI HELPERS -------------------------
async def openai_chat(messages, model="gpt-4o-mini", temperature=0.2, max_tokens=500, retries=2):
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
    trimmed_article = (full_article or "")[:1800]
    prompt = (
        "Genera campos para un post de Telegram. Reglas ESTRICTAS:\n"
        "1) Titular en español, 70–110 caracteres, informativo y específico, sin comillas ni emojis. Primera letra en mayúscula.\n"
        "2) Cuerpo: 1–2 oraciones, 220–320 caracteres. Empieza por el dato principal (qué/quién/dónde y cifra). "
        "Nada de muletillas ('Este lunes', 'En el marco de', 'Cabe señalar'). Sin enlaces. "
        "NO repitas el título: aporta un dato nuevo, avance, cifra o consecuencia.\n"
        "3) 1–3 etiquetas temáticas.\n"
        "4) Responde SOLO JSON sin ``` con formato {\"title\":\"...\",\"body\":\"...\",\"tags\":\"#...\"}\n\n"
        f"Título origen: {source_title}\n\nTexto fuente:\n{trimmed_article}"
    )
    resp = await openai_chat(model="gpt-4o-mini",
                             messages=[{"role":"user","content":prompt}],
                             temperature=0.2, max_tokens=520)
    raw = (resp.choices[0].message.content or "").strip()
    jtxt = extract_json_obj(raw) or raw

    title, body, tags = source_title, "", ""
    try:
        data = json.loads(jtxt)
        title = re.sub(r'[\"“”«»]+', '', str(data.get("title","")).strip())
        body  = str(data.get("body","")).strip()
        tags  = str(data.get("tags","")).strip()
    except Exception as e:
        logging.warning(f"JSON parse fallback: {e}; raw={raw[:200]}")

    if not title or len(title) < 40:
        fp = first_paragraph(full_article)
        title = (fp[:100] + "…") if fp else (source_title[:100] or "Actualidad")

    body = re.sub(r'\s+', ' ', body)[:340]
    tags = normalize_hashtags(tags, limit=3)
    return {"title": title, "body": body, "tags": tags}

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

# ---------------------- EVENT KEY ----------------------
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

# ------------------------- EVENTS CLUSTER (48h) -----------------
def now_ts() -> int:
    return int(time.time())

def cleanup_event_cluster(cluster: dict):
    cutoff = now_ts() - EVENT_CLUSTER_TTL_SEC
    dead = [k for k, v in cluster.items() if v.get("ts", 0) < cutoff]
    for k in dead:
        cluster.pop(k, None)

def is_same_event_recent(new_slug: str, cluster: dict, sim_ratio: float = 0.90) -> bool:
    if new_slug in cluster:
        if now_ts() - cluster[new_slug].get("ts", 0) <= EVENT_CLUSTER_TTL_SEC:
            return True
    for old in cluster.keys():
        if difflib.SequenceMatcher(None, new_slug, old).ratio() >= sim_ratio:
            if now_ts() - cluster[old].get("ts", 0) <= EVENT_CLUSTER_TTL_SEC:
                return True
    return False

def register_event(new_slug: str, url: str, title: str, cluster: dict):
    cluster[new_slug] = {"ts": now_ts(), "url": url, "title": title}

def titles_near_dup(a: str, b: str, ratio: float = 0.88) -> bool:
    a = normalize_title(a)
    b = normalize_title(b)
    if not a or not b:
        return False
    if a == b:
        return True
    return difflib.SequenceMatcher(None, a, b).ratio() >= ratio

# ---------------------- IMAGES (умный выбор) --------------------
BAD_IMG_HINTS = ["logo","icon","placeholder","default","sprite","avatar","amp","mini","thumb","svg"]
PREF_EXT = (".jpg",".jpeg",".png",".webp")
AVOID_EXT = (".gif",".svg",".ico")

def _score_url_quality(u: str) -> int:
    url = u.lower()
    score = 0
    if url.endswith(PREF_EXT): score += 15
    if url.endswith(AVOID_EXT): score -= 50
    m = re.search(r'(?:(?:w|width|ancho)=|/)(\d{3,4})(?:[x_](\d{3,4}))?', url)
    if m:
        w = int(m.group(1))
        h = int(m.group(2)) if m.group(2) else w
        score += min(w, h) // 10
    if any(hint in url for hint in BAD_IMG_HINTS): score -= 30
    if "og:image" in url or "twitter" in url: score += 10
    if url.startswith("data:") or len(url) < 10: score -= 100
    return score

def _gather_entry_images(entry) -> list[str]:
    out = []
    try:
        mc = entry.get("media_content", [])
        for it in mc:
            if it.get("url"): out.append(it["url"])
    except Exception: pass
    try:
        mt = entry.get("media_thumbnail", [])
        for it in mt:
            if it.get("url"): out.append(it["url"])
    except Exception: pass
    for l in entry.get("links", []):
        if l.get("type", "").startswith("image/") and l.get("href"):
            out.append(l["href"])
    for field in ("summary", "summary_detail"):
        html_blob = entry.get(field, "") or ""
        for m in re.finditer(r'<img[^>]+src=["\']([^"\']+)["\']', str(html_blob), re.I):
            out.append(m.group(1))
    return list(dict.fromkeys(out))

def _gather_html_images(page_html: str) -> list[str]:
    if not page_html: return []
    html_norm = page_html
    urls = []
    for m in re.finditer(r'<meta[^>]+property=["\']og:image["\'][^>]+content=["\']([^"\']+)["\']', html_norm, re.I):
        urls.append(m.group(1) + "?og:image")
    for m in re.finditer(r'<meta[^>]+name=["\']twitter:image(?::src)?["\'][^>]+content=["\']([^"\']+)["\']', html_norm, re.I):
        urls.append(m.group(1) + "?twitter:image")
    for m in re.finditer(r'<img[^>]+src=["\']([^"\']+)["\']', html_norm, re.I):
        urls.append(m.group(1))
    return list(dict.fromkeys(urls))

def extract_image(entry, page_html: str = "") -> str:
    candidates = _gather_entry_images(entry) + _gather_html_images(page_html)
    normed = []
    for u in candidates:
        if not u: continue
        if u.startswith("//"):
            u = "https:" + u
        normed.append(u)
    normed = list(dict.fromkeys(normed))
    normed.sort(key=_score_url_quality, reverse=True)
    for u in normed:
        lu = u.lower()
        if any(lu.endswith(ext) for ext in AVOID_EXT):
            continue
        if any(h in lu for h in BAD_IMG_HINTS):
            continue
        if not (lu.endswith((".jpg",".jpeg",".png",".webp"))):
            continue
        return u
    return ""

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
    global published_titles, seen_urls, EVENT_FPS, EVENT_KEYS, EVENT_CLUSTER, last_titles_raw

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

            full_article, page_html = get_full_article(clean_url)
            if not full_article:
                full_article = getattr(entry, "summary", "") or ""
            if len(full_article.split()) < 80:
                continue

            # === Жёсткий дедуп по событию ===
            fp_first_para = first_paragraph(full_article)
            event_key = await make_event_key(title, fp_first_para)
            if is_event_key_dup(event_key, EVENT_KEYS):
                continue

            # === Кластер 48ч ===
            cleanup_event_cluster(EVENT_CLUSTER)
            if is_same_event_recent(event_key, EVENT_CLUSTER, sim_ratio=0.90):
                continue

            # === simhash дедуп ===
            fp = make_event_fingerprint(title, fp_first_para)
            if fp:
                if any(hamming(fp, old) <= HAMMING_THRESHOLD_DUP for old in EVENT_FPS):
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

            # === GPT: заголовок/текст/теги
            try:
                g = await generate_full_post_with_gpt(title, full_article)
                gpt_title = g["title"]
                body = g["body"]
                tags = g["tags"]
            except Exception as e:
                logging.error(f"OpenAI post generation error: {e}")
                await notify_admin(f"❌ OpenAI error: {e}")
                continue

            # Быстрый анти-дубль по заголовку
            if any(titles_near_dup(gpt_title, old) for old in last_titles_raw[-120:]):
                continue

            title_html = f"<b>{safe_html_text(gpt_title)}</b>"
            body = safe_html_text(body)
            body = drop_duplicate_title(title_html, body)

            # дешёвый Jaccard по телу поста
            if is_jaccard_dup(body):
                continue

            # Вставляем скрытую ссылку в ключевое слово
            body = mask_link_in_body(body, clean_url)

            # Эмодзи
            emoji = final_emoji_deterministic(gpt_title + " " + body)

            # Хвост: только хэштеги и подпись канала; каждый блок отдельным абзацем
            tail_parts = []
            if tags:
                tail_parts.append(tags.lower())
            tail_parts.append(CHANNEL_SIGNATURE)
            tail = "\n\n".join(tail_parts)

            image_url = extract_image(entry, page_html)
            head = f"{emoji} {title_html}\n\n"

            if image_url:
                budget = 1024 - len(head) - len("\n\n") - len(tail)
                trimmed_body = body[:max(0, budget)]
                payload = head + trimmed_body + "\n\n" + tail
            else:
                payload = head + body + "\n\n" + tail

            try:
                for channel in CHANNEL_IDS:
                    await send_message_or_photo(channel, image_url, payload)

                # --- кэши
                seen_urls.add(clean_url)
                published_titles.add(normalize_title(gpt_title) or norm_title)
                if fp:
                    EVENT_FPS.append(fp)
                EVENT_KEYS.append(event_key)
                register_event(event_key, clean_url, gpt_title, EVENT_CLUSTER)

                save_set(CACHE_URLS, seen_urls)
                save_set(CACHE_TITLES, published_titles)
                save_fps(CACHE_FPS, EVENT_FPS)
                save_list(CACHE_EVENT_KEYS, list(EVENT_KEYS))
                save_dict(CACHE_EVENT_CLUSTER, EVENT_CLUSTER)
                last_titles_raw.append(gpt_title)
                save_list(CACHE_LAST_TITLES, last_titles_raw, maxlen=300)

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
