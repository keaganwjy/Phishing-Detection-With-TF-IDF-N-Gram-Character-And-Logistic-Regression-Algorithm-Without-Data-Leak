# Phishing Detection With TF-IDF N-Gram Character And Logistic Regression Algorithm Without Data Leak

Aplikasi berbasis machine learning yang memeriksa apakah sebuah URL mirip pola phishing, lengkap dengan alasan dan saran tindakan.

## Struktur folder
```
phishing_detection/
├── data/                       # taruh URL_dataset.csv dan Phishing_URLs.csv di sini
├── 01_EDA_Preprocessing.ipynb  # pemahaman data, EDA, fitur  -> data/data_olahan.parquet
├── 02_Modeling_Evaluation.ipynb# pembagian data, model, evaluasi, error analysis, uji fungsi -> models/
├── url_features.py             # normalisasi URL + 40 fitur (dipakai notebook DAN aplikasi)
├── predictor.py                # validasi input + prediksi + penjelasan
├── app.py                      # aplikasi Streamlit
├── models/                     # model_phishing.joblib + model_phishing_meta.json (hasil Notebook 2)
├── reports/                    # gambar & tabel hasil untuk laporan
└── requirements.txt
```

## Cara menjalankan
```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 1) Letakkan dua file CSV di folder data/
# 2) Jalankan notebook BERURUTAN dari folder ini (agar path relatif benar):
jupyter notebook                   # buka 01_..., Run All; lalu 02_..., Run All
# 3) Jalankan aplikasi
streamlit run app.py
```
> Jika versi scikit-learn/xgboost di komputermu berbeda dari yang tertulis di `models/model_phishing_meta.json`, jalankan ulang Notebook 2 agar model dibuat dengan versi milikmu.

## Sumber data
- Dataset 1: Phishing URL Dataset (Mendeley Data, `vfszbj9b36`), berkas `URL_dataset.csv`: 450.176 URL berlabel `legitimate`/`phishing`.
- Dataset 2: `Phishing_URLs.csv`: 54.807 URL phishing (tanpa legitimate).
- Rujukan metode: Shahrivari et al., *Phishing Detection Using Machine Learning Techniques*, arXiv:2009.11116.
