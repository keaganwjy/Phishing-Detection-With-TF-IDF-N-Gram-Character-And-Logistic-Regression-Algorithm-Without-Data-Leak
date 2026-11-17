"""
Lapisan antara MODEL dan APLIKASI. Berisi:
  - validasi_input()  : menolak input yang bukan URL (uji input tidak valid)
  - muat_model()      : memuat pipeline terlatih + metadata (threshold, metrik)
  - prediksi()        : memanggil model, lalu menyusun hasil + penjelasan + saran

Dipisahkan dari app.py agar bisa diuji langsung di notebook tanpa membuka Streamlit.
"""
import json
import re
from pathlib import Path

import joblib

from url_features import daftar_red_flag, normalisasi_url

DIR_MODEL = Path(__file__).parent / "models"
BATAS_PANJANG = 2048
_SKEMA_DITOLAK = ("javascript:", "data:", "file:", "vbscript:", "mailto:", "tel:", "about:", "blob:")
_RE_HOST = re.compile(r"^(\[[0-9a-fA-F:]+\]|[\w.\-]+)$", re.UNICODE)
_RE_IP = re.compile(r"^(\d{1,3}\.){3}\d{1,3}$")


def validasi_input(teks):
    """Return (valid: bool, pesan: str). Pesan menjelaskan kenapa input ditolak."""
    if teks is None or not isinstance(teks, str):
        return False, "Input harus berupa teks."
    t = teks.strip()
    if not t:
        return False, "URL masih kosong. Tempel tautan yang ingin diperiksa."
    if len(t) > BATAS_PANJANG:
        return False, f"URL terlalu panjang ({len(t)} karakter; batas {BATAS_PANJANG})."
    if re.search(r"\s", t):
        return False, "URL tidak boleh mengandung spasi. Pastikan hanya satu tautan yang ditempel."
    if t.lower().startswith(_SKEMA_DITOLAK):
        return False, "Hanya tautan web (http/https) yang dapat diperiksa."
    m = re.match(r"^([a-zA-Z][a-zA-Z0-9+.\-]*)://", t)
    if m and m.group(1).lower() not in ("http", "https", "ftp", "httpss"):
        return False, f"Skema '{m.group(1)}://' tidak didukung. Gunakan tautan http atau https."
    n = normalisasi_url(t)
    host = re.split(r"[/?#]", n, maxsplit=1)[0]
    host = host.rsplit("@", 1)[-1]
    host_tanpa_port = re.sub(r":\d+$", "", host)
    if not host_tanpa_port or not _RE_HOST.match(host_tanpa_port):
        return False, "Bagian alamat situs (domain) tidak valid."
    if "." not in host_tanpa_port and not host_tanpa_port.startswith("["):
        return False, "Domain tidak lengkap (tidak ada titik, mis. 'contoh.com')."
    if host_tanpa_port.endswith(".") or ".." in host_tanpa_port:
        return False, "Format domain tidak valid."
    if _RE_IP.match(host_tanpa_port) and any(int(x) > 255 for x in host_tanpa_port.split(".")):
        return False, "Alamat IP tidak valid."
    return True, "OK"


# Layanan sah yang juga sering disalahgunakan penipu. Model bisa menganggapnya mencurigakan
# karena data latih memuat banyak phishing di platform ini. Kita beri catatan konteks (bukan mengubah prediksi).
LAYANAN_DISALAHGUNAKAN = (
    "docs.google.com", "drive.google.com", "forms.gle", "sites.google.com", "github.com", "github.io",
    "notion.site", "weebly.com", "weeblysite.com", "wixsite.com", "blogspot.com", "pages.dev", "web.app",
    "firebaseapp.com", "netlify.app", "vercel.app", "herokuapp.com", "webwave.dev", "cloudflare-ipfs.com",
)


def _catatan_konteks(url):
    host = re.split(r"[/?#]", normalisasi_url(url), maxsplit=1)[0].split("@")[-1].split(":")[0]
    for d in LAYANAN_DISALAHGUNAKAN:
        if host == d or host.endswith("." + d):
            return (f"Alamat ini berada di layanan umum ({d}) yang dipakai banyak orang, termasuk penipu. "
                    "Hasil model dapat keliru; periksa siapa pengirim tautan dan isi halamannya sebelum memasukkan data.")
    return None


_CACHE = {}


def muat_model(nama="model_phishing"):
    """Muat pipeline & metadata sekali saja (di-cache)."""
    if nama not in _CACHE:
        pipe = joblib.load(DIR_MODEL / f"{nama}.joblib")
        meta = json.loads((DIR_MODEL / f"{nama}_meta.json").read_text(encoding="utf-8"))
        _CACHE[nama] = (pipe, meta)
    return _CACHE[nama]


def _saran(status):
    if status == "PHISHING":
        return [
            "Jangan klik tautan ini dan jangan memasukkan kata sandi, OTP, atau data kartu.",
            "Jika sudah terlanjur membuka atau mengisi data, segera ganti kata sandi akun terkait dan aktifkan verifikasi dua langkah.",
            "Jika tautan mengaku dari bank/marketplace, buka aplikasi atau situs resminya langsung (ketik sendiri alamatnya), jangan lewat tautan tersebut.",
            "Laporkan tautan ke penyedia layanan terkait atau ke aduannomor.id / Kominfo.",
        ]
    if status == "MENCURIGAKAN":
        return [
            "Model menemukan beberapa ciri yang mirip phishing. Hindari memasukkan data sensitif di situs ini.",
            "Periksa ejaan domain dengan teliti dan pastikan tautan berasal dari sumber yang kamu percaya.",
            "Jika ragu, buka situs resminya secara manual lewat mesin pencari atau aplikasi resmi.",
        ]
    return [
        "Tidak ada indikasi kuat phishing dari pola URL, tetapi ini BUKAN jaminan keamanan.",
        "Tetap waspada: periksa ejaan domain dan jangan membagikan OTP/kata sandi lewat tautan yang dikirim orang lain.",
    ]


def prediksi(teks):
    """Prediksi satu URL. Selalu mengembalikan dict (tidak melempar error ke UI)."""
    ok, pesan = validasi_input(teks)
    if not ok:
        return {"valid": False, "pesan": pesan}
    pipe, meta = muat_model()
    url = teks.strip()
    p = float(pipe.predict_proba([url])[0, 1])
    thr = meta["threshold"]
    if p >= thr:
        status = "PHISHING"
    elif p >= 0.5 * thr:
        status = "MENCURIGAKAN"
    else:
        status = "TIDAK TERDETEKSI PHISHING"
    return {
        "valid": True,
        "url": url,
        "url_dinormalisasi": normalisasi_url(url),
        "probabilitas_phishing": p,
        "threshold": thr,
        "status": status,
        "red_flags": daftar_red_flag(url),
        "catatan_konteks": _catatan_konteks(url),
        "saran": _saran("PHISHING" if status == "PHISHING" else "MENCURIGAKAN" if status == "MENCURIGAKAN" else "AMAN"),
        "model": meta["nama_model"],
    }
