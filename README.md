# AI Leaf Doctor 🍂
### Sistem Identifikasi & Estimasi Luas Area Infeksi Penyakit Tanaman Apel Berbasis *Instance Segmentation*

Aplikasi berbasis web ini dikembangkan menggunakan **YOLOv8-Segmentation** (Ultralytics) sebagai *backend* kecerdasan buatan dan **Streamlit** sebagai antarmuka *frontend*. Sistem ini dirancang untuk mendeteksi jenis penyakit pada daun apel (*Apple Scab*, *Apple Rust*, dan *Black Rot*) sekaligus menghitung persentase luas area infeksi secara otomatis berbasis piksel spasial untuk memberikan rekomendasi tindakan kepada petani.

Proyek ini dibuat untuk memenuhi **Tugas Akhir UAS Mata Kuliah Advanced Computer Vision** - Program Studi Teknik Informatika, Universitas Pamulang (2026).

---

## 🚀 Fitur Utama
1. **Instance Segmentation:** Mendeteksi letak bercak penyakit dan menggambarkan poligon masker penanda secara presisi piksel demi piksel.
2. **Kalkulator Luas Spasial:** Menghitung persentase estimasi luas area infeksi tanaman terhadap keseluruhan dimensi frame foto secara otomatis.
3. **Sistem Pakar Sederhana:** Menyajikan kotak notifikasi rekomendasi tindakan penanganan (Hijau/Kuning/Merah) berdasarkan tingkat keparahan infeksi daun.

---

## 📂 Struktur Direktori Proyek
```text
proyek-uas-cv/
│
├── env/                 # Folder Virtual Environment Python (diabaikan oleh Git)
├── best.pt              # File Bobot Biner Model YOLOv8 hasil training Colab
├── app.py               # Source Code utama aplikasi web Streamlit
├── README.md            # Dokumentasi proyek GitHub
└── .gitignore           # Konfigurasi pengabaian file Git
```

---

## 💻 Panduan Instalasi & Cara Menjalankan

Berikut adalah panduan menjalankan proyek ini secara lokal pada perangkat Anda (dioptimalkan untuk macOS & Windows).

### 1. Kloning Repositori (Jika di-upload ke GitHub)
```bash
git clone https://github.com/username-anda/nama-repo-anda.git
cd nama-repo-anda
```
*Atau jika Anda langsung membuat foldernya secara lokal, masuk ke folder melalui Terminal/CMD:*
```bash
cd path/to/proyek-uas-cv
```

### 2. Membuat & Mengaktifkan Virtual Environment
* **macOS / Linux:**
  ```bash
  python3 -m venv env
  source env/bin/activate
  ```
* **Windows (Command Prompt):**
  ```bash
  python -m venv env
  env\Scripts\activate
  ```

### 3. Instalasi Dependency (Library)
Pastikan virtual environment Anda telah aktif (ditandai dengan tulisan `(env)` pada terminal), kemudian instal pustaka pendukung berikut:
```bash
pip install streamlit ultralytics pillow "numpy<2" "opencv-python<4.10"
```

### 4. Menjalankan Aplikasi Web
Pastikan file bobot `best.pt` sudah diletakkan di dalam folder yang sama dengan file `app.py`. Jalankan perintah berikut:
```bash
streamlit run app.py
```
Aplikasi web otomatis akan terbuka pada peramban (*browser*) Anda di alamat `http://localhost:8501`.

---

## 📊 Dataset & Metrik Evaluasi
* **Sumber Data:** *Annotated Apple Leaf Disease Dataset* (Kaggle & Roboflow Universe).
* **Arsitektur Model:** YOLOv8 Nano Segmentation (`yolov8n-seg.pt`).
* **Metrik Evaluasi Latihan:** Penurunan fungsi *loss* (*box_loss*, *seg_loss*, *cls_loss*) dan peningkatan nilai *Mean Average Precision* (mAP50) untuk performa segmentasi topologi geometri bercak daun.

---

## 📝 Anggota Tim / Pengembang
* **Nama:** [Isi Nama Anda]
* **NIM:** [Isi NIM Anda]
* **Dosen Pengampu:** Dr. Arya Adhyaksa Waskita
