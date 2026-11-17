import pandas as pd
import streamlit as st

from predictor import muat_model, prediksi, validasi_input

st.set_page_config(page_title="Cek Tautan — Deteksi Phishing", page_icon="🛡️", layout="centered")

WARNA = {"PHISHING": "#b91c1c", "MENCURIGAKAN": "#b45309", "TIDAK TERDETEKSI PHISHING": "#15803d"}
IKON = {"PHISHING": "⛔", "MENCURIGAKAN": "⚠️", "TIDAK TERDETEKSI PHISHING": "✅"}
LEVEL = {"tinggi": "🔴 Tinggi", "sedang": "🟠 Sedang", "rendah": "🟡 Rendah"}

# Contoh simulasi (aman: tidak mengarah ke situs sungguhan; hanya string untuk didemokan)
CONTOH = {
    "Situs resmi (Wikipedia)": "https://id.wikipedia.org/wiki/Phishing",
    "Phishing: IP + login": "http://192.168.10.5/secure-login/verify-account.php?user=1",
    "Phishing: tiru PayPal": "http://paypal.com.account-verify.tk/signin",
    "Phishing: tiru bank": "http://bca-klikbca.verifikasi-akun.xyz/login.php",
}


@st.cache_resource
def _siap():
    return muat_model()


try:
    _, META = _siap()
except Exception as e:  # model belum dibuat / versi library berbeda
    st.error("Model belum ditemukan atau gagal dimuat. Jalankan Notebook 2 sampai selesai, lalu muat ulang halaman ini.")
    st.caption(f"Detail teknis: {e}")
    st.stop()

# ----------------------------- Sidebar -----------------------------
with st.sidebar:
    st.header("Tentang aplikasi")
    st.write("Memeriksa apakah sebuah tautan **mirip pola phishing** berdasarkan teks URL-nya. "
             "Tidak membuka tautan dan tidak mengirim data ke pihak lain.")
    m = META["metrik_uji"]
    st.subheader("Performa model (data uji)")
    c1, c2 = st.columns(2)
    c1.metric("Recall phishing", f"{m['recall']:.1%}")
    c2.metric("Precision", f"{m['precision']:.1%}")
    c1.metric("F1", f"{m['f1']:.3f}")
    c2.metric("Akurasi", f"{m['accuracy']:.1%}")
    st.caption(f"Model: {META['nama_model']} • ambang batas {META['threshold']:.2f}")
    st.subheader("Keterbatasan")
    st.markdown(
        "- Hanya membaca teks URL, bukan isi halaman.\n"
        "- Situs baru/bersih bisa lolos; situs sah di platform umum bisa salah ditandai.\n"
        "- **Bukan jaminan keamanan.** Tetap waspada pada permintaan OTP/kata sandi."
    )

# ----------------------------- Halaman utama -----------------------------
st.title("🛡️ Cek Tautan")
st.write("Tempel tautan yang kamu terima lewat chat, SMS, atau email, lalu periksa sebelum membukanya.")

tab1, tab2 = st.tabs(["Periksa satu tautan", "Periksa banyak tautan"])

with tab1:
    if "url_input" not in st.session_state:
        st.session_state["url_input"] = ""
    st.caption("Coba contoh simulasi:")
    kolom = st.columns(len(CONTOH))
    for k, (nama, url) in zip(kolom, CONTOH.items()):
        if k.button(nama):
            st.session_state["url_input"] = url

    url = st.text_input("Alamat tautan (URL)", key="url_input", placeholder="contoh: https://situs-bank.co.id/login")
    if st.button("Periksa", type="primary"):
        hasil = prediksi(url)
        if not hasil["valid"]:
            st.warning(f"Input tidak dapat diperiksa: {hasil['pesan']}")
        else:
            s, p = hasil["status"], hasil["probabilitas_phishing"]
            st.markdown(
                f"<div style='padding:16px 20px;border-radius:10px;background:{WARNA[s]}1a;border-left:6px solid {WARNA[s]}'>"
                f"<div style='font-size:1.4rem;font-weight:700;color:{WARNA[s]}'>{IKON[s]} {s}</div>"
                f"<div>Skor risiko phishing: <b>{p:.1%}</b> (batas peringatan {hasil['threshold']:.0%})</div></div>",
                unsafe_allow_html=True,
            )
            st.progress(min(max(p, 0.0), 1.0))
            if hasil.get("catatan_konteks"):
                st.info(hasil["catatan_konteks"])

            st.subheader("Mengapa?")
            if hasil["red_flags"]:
                for tingkat, teks in hasil["red_flags"]:
                    st.markdown(f"- {LEVEL[tingkat]} — {teks}")
            else:
                st.write("Tidak ada tanda bahaya yang jelas dari aturan sederhana (IP, '@', kata sensitif, peniruan merek, dst.). "
                         "Skor di atas berasal dari pola karakter pada URL yang dipelajari model.")
            st.caption("Daftar ini penjelasan sederhana berbasis aturan; keputusan akhir tetap berasal dari skor model.")

            st.subheader("Saran tindakan")
            for t in hasil["saran"]:
                st.markdown(f"- {t}")

            with st.expander("Detail teknis"):
                st.write("URL setelah normalisasi (skema `http(s)://` dan `www.` dibuang):")
                st.code(hasil["url_dinormalisasi"], language=None)
                st.json({"probabilitas_phishing": round(p, 4), "threshold": round(hasil["threshold"], 4), "model": hasil["model"]})

with tab2:
    st.write("Tempel beberapa tautan, **satu per baris** (maksimal 200).")
    teks = st.text_area("Daftar tautan", height=160, key="banyak")
    if st.button("Periksa semua"):
        baris = [b.strip() for b in teks.splitlines() if b.strip()]
        if not baris:
            st.warning("Belum ada tautan.")
        else:
            if len(baris) > 200:
                st.info("Hanya 200 tautan pertama yang diperiksa.")
                baris = baris[:200]
            out = []
            for b in baris:
                r = prediksi(b)
                if r["valid"]:
                    out.append({"url": b, "status": r["status"], "skor_phishing": round(r["probabilitas_phishing"], 3)})
                else:
                    out.append({"url": b, "status": "INPUT TIDAK VALID", "skor_phishing": None})
            tabel = pd.DataFrame(out)
            st.dataframe(tabel, hide_index=True)
            st.download_button("Unduh hasil (CSV)", tabel.to_csv(index=False).encode("utf-8"), "hasil_cek_tautan.csv", "text/csv")

st.divider()
st.caption("Proyek UAS Keamanan Data • Model dilatih dari data publik (Mendeley) • Hasil bersifat indikatif, bukan jaminan.")