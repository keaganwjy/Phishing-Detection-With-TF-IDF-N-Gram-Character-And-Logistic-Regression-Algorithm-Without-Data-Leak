"""
Isi modul:
  1. normalisasi_url()      -> membersihkan URL (buang skema & 'www.')
  2. ekstrak_fitur()        -> mengubah satu URL menjadi dict fitur numerik
  3. buat_matriks_fitur()   -> versi untuk banyak URL (DataFrame)
  4. daftar_red_flag()      -> alasan yang bisa dibaca manusia (untuk aplikasi)
"""
import math
import re
from collections import Counter

import numpy as np
import pandas as pd
import tldextract

# Ekstraktor domain memakai daftar suffix bawaan (offline, tanpa download)
_EXT = tldextract.TLDExtract(suffix_list_urls=(), cache_dir=None)

# --------------------------------------------------------------------------
# Daftar pengetahuan domain (silakan kamu tambah sendiri sebagai latihan)
# --------------------------------------------------------------------------
KATA_SENSITIF = [
    "login", "signin", "sign-in", "verify", "verification", "secure", "security",
    "account", "update", "confirm", "banking", "bank", "password", "passwd",
    "webscr", "wallet", "invoice", "payment", "billing", "suspend", "unlock",
    "recover", "support", "helpdesk", "credential", "authenticate",
]
MEREK_TARGET = [
    "paypal", "google", "facebook", "apple", "microsoft", "amazon", "netflix",
    "instagram", "whatsapp", "linkedin", "dropbox", "ebay", "bca", "mandiri",
    "bni", "bri", "gopay", "ovo", "dana", "shopee", "tokopedia",
]
PEMENDEK_URL = {
    "bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly", "is.gd", "buff.ly",
    "cutt.ly", "rebrand.ly", "shorturl.at", "tiny.cc", "rb.gy",
}
TLD_RISIKO_TINGGI = {
    "tk", "ml", "ga", "cf", "gq", "xyz", "top", "club", "work", "click",
    "link", "info", "icu", "buzz", "monster", "cyou", "rest", "zip",
}

_RE_SKEMA = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://")
_RE_IP = re.compile(r"^(\d{1,3}\.){3}\d{1,3}$")
_RE_IP_HEX = re.compile(r"^0x[0-9a-fA-F]+$")


# --------------------------------------------------------------------------
# 1. NORMALISASI
# --------------------------------------------------------------------------
def normalisasi_url(url: str) -> str:
    """Buang spasi, skema (http/https/ftp...) dan awalan 'www.'.

    ALASAN (temuan EDA): pada dataset, 100% URL legitimate memakai https dan
    99,8% diawali 'www.', sedangkan phishing tidak. Itu adalah artefak cara
    data dikumpulkan, BUKAN sifat asli phishing. Jika dibiarkan, model akan
    'menyontek' dari https/www dan gagal pada situs phishing ber-https.
    """
    u = str(url).strip()
    # Artefak pengumpulan data (hasil EDA): '&amp;' (HTML-escape) muncul di ~5% phishing tetapi
    # 0,02% legitimate, dan akhiran '...' (URL terpotong) hanya ada pada phishing. Keduanya
    # bukan sifat phishing sebenarnya, jadi dinormalkan agar tidak dijadikan jalan pintas.
    u = u.replace("&amp;", "&")
    u = re.sub(r"(\.\.\.|\u2026)+$", "", u)
    u = _RE_SKEMA.sub("", u)
    # host tidak peka huruf besar/kecil -> samakan; path/query dibiarkan apa adanya
    m = re.match(r"^([^/?#]*)(.*)$", u, flags=re.DOTALL)
    u = m.group(1).lower() + m.group(2)
    u = re.sub(r"^www\d{0,3}\.", "", u)
    return u


def normalisasi_batch(urls):
    """Versi banyak-URL dari normalisasi_url (dipakai di Pipeline TF-IDF)."""
    return [normalisasi_url(u) for u in urls]


def _pecah(u_norm: str):
    """Pisahkan URL ternormalisasi menjadi host, path, query."""
    # buang fragment, pisahkan query
    u = u_norm.split("#", 1)[0]
    awal, _, query = u.partition("?")
    host_port, _, path = awal.partition("/")
    path = "/" + path if path or "/" in awal else ""
    # buang userinfo (user:pass@host)
    userinfo = ""
    if "@" in host_port:
        userinfo, host_port = host_port.rsplit("@", 1)
    host, _, port = host_port.partition(":")
    return host.lower(), path, query, port, userinfo


def _entropi(teks: str) -> float:
    """Entropi Shannon: seberapa 'acak' sebuah string (domain DGA cenderung tinggi)."""
    if not teks:
        return 0.0
    n = len(teks)
    return -sum((c / n) * math.log2(c / n) for c in Counter(teks).values())


# --------------------------------------------------------------------------
# 2. EKSTRAKSI FITUR (satu URL)
# --------------------------------------------------------------------------
def ekstrak_fitur(url: str) -> dict:
    u = normalisasi_url(url)
    host, path, query, port, userinfo = _pecah(u)
    ex = _EXT(host)
    domain, suffix, subdomain = ex.domain, ex.suffix, ex.subdomain
    u_low = u.lower()

    host_adalah_ip = bool(_RE_IP.match(host) or _RE_IP_HEX.match(host))
    huruf = sum(c.isalpha() for c in u)
    angka = sum(c.isdigit() for c in u)
    panjang = max(len(u), 1)
    reg_domain = f"{domain}.{suffix}" if suffix else domain
    # merek yang muncul di URL tetapi BUKAN domain utamanya (indikasi peniruan)
    sisa = u_low.replace(reg_domain.lower(), "") if reg_domain else u_low
    merek_palsu = int(any(m in sisa for m in MEREK_TARGET) and not any(m in domain for m in MEREK_TARGET))

    f = {
        # --- panjang ---
        "len_url": len(u),
        "len_host": len(host),
        "len_path": len(path),
        "len_query": len(query),
        "len_domain": len(domain),
        # --- karakter khusus ---
        "n_titik": u.count("."),
        "n_hyphen": u.count("-"),
        "n_underscore": u.count("_"),
        "n_slash": u.count("/"),
        "n_tanya": u.count("?"),
        "n_sama_dengan": u.count("="),
        "n_dan": u.count("&"),
        "n_at": u.count("@"),
        "n_persen": u.count("%"),
        "n_hyphen_host": host.count("-"),
        "n_titik_host": host.count("."),
        # --- komposisi ---
        "n_angka": angka,
        "rasio_angka": angka / panjang,
        "rasio_huruf": huruf / panjang,
        "n_huruf_besar": sum(c.isupper() for c in u),
        "angka_di_host": sum(c.isdigit() for c in host),
        "digit_run_terpanjang": max((len(m) for m in re.findall(r"\d+", u)), default=0),
        # --- struktur host ---
        "host_adalah_ip": int(host_adalah_ip),
        "n_subdomain": len([s for s in subdomain.split(".") if s]),
        "ada_port": int(bool(port)),
        "ada_userinfo": int(bool(userinfo)),
        "panjang_tld": len(suffix),
        "tld_risiko_tinggi": int(suffix.split(".")[-1] in TLD_RISIKO_TINGGI) if suffix else 0,
        "pemendek_url": int(reg_domain.lower() in PEMENDEK_URL),
        # --- path ---
        "kedalaman_path": len([p for p in path.split("/") if p]),
        "double_slash_path": int("//" in path),
        "ekstensi_php": int(".php" in u_low),
        "ekstensi_html": int(bool(re.search(r"\.html?($|\?)", u_low))),
        "ada_exe_zip": int(bool(re.search(r"\.(exe|zip|rar|apk|scr|bat)($|\?)", u_low))),
        "ada_http_di_path": int("http" in (path + query).lower()),
        # --- kata kunci ---
        "n_kata_sensitif": sum(k in u_low for k in KATA_SENSITIF),
        "kata_sensitif_di_host": int(any(k in host for k in KATA_SENSITIF)),
        "merek_palsu": merek_palsu,
        # --- keacakan ---
        "entropi_host": _entropi(host),
        "entropi_path": _entropi(path + query),
    }
    return f


NAMA_FITUR = list(ekstrak_fitur("example.com/a?b=1").keys())


# --------------------------------------------------------------------------
# 3. EKSTRAKSI FITUR (banyak URL) – dipakai sebagai transformer sklearn
# --------------------------------------------------------------------------
def buat_matriks_fitur(urls) -> pd.DataFrame:
    """Terima list/Series URL -> DataFrame fitur (urutan kolom = NAMA_FITUR)."""
    baris = [ekstrak_fitur(u) for u in urls]
    return pd.DataFrame(baris, columns=NAMA_FITUR).astype(float)


def domain_terdaftar(url: str) -> str:
    """Domain utama (mis. 'example.co.uk'), dipakai untuk GROUP split agar
    URL dari domain yang sama tidak terpisah di train dan test."""
    host, *_ = _pecah(normalisasi_url(url))
    ex = _EXT(host)
    if _RE_IP.match(host):
        return host
    return f"{ex.domain}.{ex.suffix}".strip(".") or host


# --------------------------------------------------------------------------
# 4. ALASAN YANG BISA DIBACA MANUSIA (untuk aplikasi)
# --------------------------------------------------------------------------
def daftar_red_flag(url: str) -> list:
    """Kembalikan daftar (tingkat, penjelasan) berdasarkan aturan transparan.
    Ini TIDAK menggantikan model, hanya membantu pengguna memahami hasilnya."""
    f = ekstrak_fitur(url)
    bendera = []
    if f["host_adalah_ip"]:
        bendera.append(("tinggi", "Alamat memakai angka IP, bukan nama domain. Situs resmi hampir tidak pernah begini."))
    if f["n_at"] > 0 or f["ada_userinfo"]:
        bendera.append(("tinggi", "Ada simbol '@' di URL. Peramban mengabaikan teks sebelum '@', trik umum untuk menyamarkan tujuan."))
    if f["merek_palsu"]:
        bendera.append(("tinggi", "Nama merek terkenal muncul di URL, tetapi bukan domain utamanya (indikasi peniruan merek)."))
    if f["pemendek_url"]:
        bendera.append(("sedang", "Memakai layanan pemendek URL sehingga tujuan asli tersembunyi."))
    if f["ada_exe_zip"]:
        bendera.append(("tinggi", "URL mengarah ke berkas yang dapat dieksekusi/arsip (.exe/.zip/.apk, dst.)."))
    if f["tld_risiko_tinggi"]:
        bendera.append(("sedang", "Ekstensi domain (TLD) sering disalahgunakan untuk situs murahan/sekali pakai."))
    if f["n_kata_sensitif"] >= 2:
        bendera.append(("sedang", f"Mengandung {int(f['n_kata_sensitif'])} kata sensitif (login/verify/secure/dst.) yang lazim dipakai untuk memancing pengguna."))
    if f["n_subdomain"] >= 3:
        bendera.append(("sedang", f"Subdomain bertingkat ({int(f['n_subdomain'])} tingkat); sering dipakai untuk membuat URL tampak resmi."))
    if f["n_hyphen_host"] >= 3:
        bendera.append(("sedang", "Banyak tanda '-' pada nama domain."))
    if f["len_url"] > 100:
        bendera.append(("rendah", f"URL sangat panjang ({int(f['len_url'])} karakter)."))
    if f["ada_http_di_path"]:
        bendera.append(("sedang", "Ada 'http' tertanam di dalam path/query (pola pengalihan/penyamaran)."))
    if f["entropi_host"] > 3.8 and f["len_host"] > 12:
        bendera.append(("rendah", "Nama domain tampak acak (entropi tinggi)."))
    return bendera
