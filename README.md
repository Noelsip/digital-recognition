# Hand Character Recognition

CNN untuk pengenalan tulisan tangan **0-9 dan A-Z** (36 karakter total) dengan multi-character segmentation. Tulis "11", "ABC", atau "HELLO123" — semua bisa dibaca.

**Dataset:** MNIST (digits) + EMNIST letters, otomatis di-download via `kagglehub`.

## 📁 Struktur Project

```
digit_recognition/
├── notebooks/
│   ├── train_model.ipynb       # Notebook training (Colab/VS Code/Jupyter)
│   └── draw_and_predict.ipynb  # Drawing inline (alternatif)
├── src/
│   ├── dataset_loader.py       # Auto-download MNIST + EMNIST
│   ├── train.py                # Training script
│   └── predict.py              # Prediction module
├── interface/
│   ├── draw_app.py             # Desktop GUI dengan multi-char + mode selector
│   └── draw_app.html           # Web interface (digits only)
├── models/                     # Output: char_model.keras
├── quickstart.py
├── requirements.txt
└── README.md
```

## 🚀 Cara Pakai (Workflow Recommended: Hybrid Colab + Lokal)

**Training di Colab** (manfaatkan GPU gratis), **drawing di laptop** (interface smooth & real-time).

### Step 1 — Training di Colab

1. Upload `notebooks/train_model.ipynb` ke [Google Colab](https://colab.research.google.com)
2. **Runtime → Change runtime type → GPU (T4)**
3. **Run all cells**. Total ~30-50 menit untuk 10 epochs (dataset gabungan ~880K sample).
4. Di cell terakhir, **uncomment** dua baris ini lalu run:
   ```python
   from google.colab import files
   files.download('../models/char_model.keras')
   ```
5. File `char_model.keras` ter-download ke laptop.

### Step 2 — Drawing di Laptop

1. Extract ZIP project ini di laptop.
<!-- 2. Taruh file `char_model.keras` (yang baru di-download) ke folder `models/` di project. -->
3. Install dependencies (yang dibutuhkan untuk drawing app saja, lebih ringan):
   ```bash
   pip install tensorflow pillow numpy scipy
   ```
4. Jalankan:
   ```bash
   python interface/draw_app.py
   ```
5. Pilih mode di atas kanvas:
   - **🔢 Digits** — kalau kamu cuma mau predict angka 0-9
   - **🔤 Letters** — kalau cuma huruf A-Z
   - **🔀 Mixed** — bisa angka dan huruf (default)
6. Tulis di kanvas. **Pisahkan tiap karakter dengan jarak yang jelas** supaya segmentation bekerja.

## 🚀 Alternatif: Semua di Laptop

Kalau kamu tidak mau pakai Colab dan punya kesabaran (training di CPU bisa 2-3 jam):

```bash
pip install -r requirements.txt
python src/train.py --epochs 10
python interface/draw_app.py
```

## 🎨 Cara Kerja Multi-Character

Model masih cuma bisa baca **1 karakter per gambar 28×28**. Yang membuat multi-char bekerja adalah **segmentasi di sisi interface**:

1. Canvas di-scan, cari kelompok piksel yang terhubung pakai `scipy.ndimage.label`
2. Tiap kelompok di-crop, di-resize ke 28×28 (format MNIST/EMNIST)
3. Diprediksi satu per satu, hasilnya digabung sesuai urutan kiri-ke-kanan

**Limitasi yang harus diketahui:**

- Tulisan **harus print/cetak** dengan jarak antar karakter yang jelas. Tulisan sambung (cursive) tidak akan bekerja karena seluruh kata terbaca sebagai 1 "blob" piksel.
- Karakter mirip (O vs 0, I vs 1 vs L, S vs 5, B vs 8) sering keliru bahkan untuk manusia. **Pakai mode selector** untuk batasi pilihan: kalau kamu lagi nulis nomor telepon, pilih mode "Digits"; kalau nama, pilih "Letters".
- Karakter yang menempel (misal "10" tapi 0-nya nyentuh 1) kemungkinan dianggap 1 karakter dan salah prediksi.

## 📊 Tentang Model

- CNN ~520K params (sedikit lebih besar dari versi MNIST-only karena 36 classes)
- Akurasi keseluruhan ~92-95% (digits ~99%, letters ~88-92% — ada huruf yang ambigu)
- Training time: ~3-5 menit/epoch di Colab GPU, ~30-45 menit/epoch di CPU
- Dataset gabungan: 60K MNIST train + 124K EMNIST letters train = 184K, ditambah test set untuk validation

## 🔧 Bonus: TensorFlow.js (untuk web interface)

```bash
pip install tensorflowjs
tensorflowjs_converter --input_format keras models/char_model.h5 interface/web_model/
```

Catatan: `draw_app.html` saat ini di-config untuk 10 classes (MNIST). Untuk pakai model 36 classes, edit JS di file tersebut (perlu tambah label mapping). Lebih praktis pakai `draw_app.py` untuk versi character recognition.

## 🐛 Troubleshooting

**"scipy belum terinstall"** — `pip install scipy`. Wajib untuk multi-character segmentation.

**"Model tidak ditemukan"** — Pastikan `char_model.keras` ada di folder `models/`. Kalau training di Colab, jangan lupa download file model.

**Akurasi huruf rendah** — Wajar untuk huruf ambigu. Pakai mode selector untuk membatasi pilihan ke kategori yang tepat.

**Karakter terdeteksi salah jumlahnya** — Tulis dengan jarak yang lebih jelas antar karakter. Stroke yang terputus dalam 1 huruf juga bisa kebaca sebagai 2 karakter — tulis dengan stroke yang nyambung.

**EMNIST download error 401** — Untuk dataset publik kagglehub harusnya bisa, tapi kalau muncul login prompt, login dulu di [kaggle.com](https://kaggle.com) di browser, atau pakai `kagglehub.login()` di Python sekali.

## 📝 Lisensi

MNIST & EMNIST adalah public domain. Kode project ini bebas dipakai.
