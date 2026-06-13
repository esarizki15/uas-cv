import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np

# ====================================================================
# 1. KONFIGURASI ANTARMUKA WEB STREAMLIT
# ====================================================================
st.set_page_config(page_title="AI Leaf Doctor", layout="centered", page_icon="🍂")

st.title("🍂 AI Leaf Doctor")
st.subheader("Sistem Identifikasi & Estimasi Luas Area Infeksi Penyakit Tanaman Apel")
st.caption("Dikembangkan untuk Tugas Akhir UAS - Mata Kuliah Advanced Computer Vision")
st.write("---")

# ====================================================================
# 2. MEMUAT MODEL AI (DENGAN CACHING AGAR CEPAT)
# ====================================================================
# Fungsi cache memastikan model hanya dimuat satu kali ke memori laptop saat web dibuka
@st.cache_resource
def load_saved_model():
    # Pastikan file 'best.pt' hasil download dari Colab diletakkan di folder yang sama dengan file app.py ini
    return YOLO("best.pt")

try:
    model = load_saved_model()
except Exception as e:
    st.error("⚠️ Gagal memuat file 'best.pt'. Pastikan file tersebut sudah diletakkan di folder proyek Anda.")

# ====================================================================
# 3. KOMPONEN UNTUK UNGGAH GAMBAR DAUN
# ====================================================================
uploaded_file = st.file_uploader("Pilih dan unggah foto daun apel yang bergejala sakit...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Membuka gambar yang diunggah menggunakan library PIL
    image = Image.open(uploaded_file)
    
    # Menampilkan gambar asli milik pengguna di halaman web
    st.image(image, caption="Gambar Daun Asli", use_container_width=True)
    
    # Menampilkan animasi loading saat AI bekerja
    with st.spinner('Sedang menganalisis penyakit dan menghitung kalkulasi luas infeksi...'):
        
        # Mengubah format gambar PIL ke format matriks NumPy BGR agar bisa diproses oleh OpenCV
        opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Jalankan proses inferensi/prediksi menggunakan model YOLOv8-Segmentation
        results = model(opencv_image)
        result = results[0]  # Mengambil hasil prediksi dari gambar pertama
        
        # FUNGSI OTOMATIS: Menggambar garis tepi poligon (masker) penyakit di atas gambar asli
        res_plotted = result.plot()
        # Mengembalikan format warna ke RGB agar bisa ditampilkan dengan benar di Streamlit
        res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
        
    # Tampilkan Gambar Hasil Segmentasi AI
    st.success("🎉 Analisis Deteksi Selesai!")
    st.image(res_rgb, caption="Hasil Segmentasi Poligon Area Infeksi Penyakit", use_container_width=True)
    
    # ====================================================================
    # 4. PROSES ESTIMASI LUAS AREA INFEKSI (FITUR UTAMA SESUAI JUDUL)
    # ====================================================================
    st.markdown("### 📊 Hasil Analisis & Estimasi Spasial")
    
    # Memeriksa apakah model menemukan adanya masker infeksi/penyakit pada daun
    if result.masks is not None:
        # Menampilkan informasi detail nama penyakit dan confidence score-nya
        boxes = result.boxes
        for box in boxes:
            class_id = int(box.cls[0])
            nama_penyakit = model.names[class_id]
            confidence = float(box.conf[0]) * 100
            st.write(f"• **Jenis Penyakit:** `{nama_penyakit}` | **Confidence Score:** `{confidence:.2f}%`")
        
        st.write("---")
        
        # ALGORITMA HITUNG PIKSEL:
        # Mengambil data matriks biner masker (0 = sehat/background, 1 = area penyakit)
        masks_data = result.masks.data.cpu().numpy()
        
        # Menggabungkan seluruh masker jika AI menemukan lebih dari satu titik bercak infeksi
        combined_mask = np.any(masks_data, axis=0)
        
        # Menghitung jumlah piksel yang bernilai True (area yang sakit)
        piksel_infeksi = np.sum(combined_mask)
        # Menghitung total seluruh piksel pada frame gambar
        total_piksel = combined_mask.size
        
        # Rumus Estimasi Luas: (Piksel Sakit / Total Piksel Gambar) x 100%
        persentase_luas_infeksi = (piksel_infeksi / total_pixels_in_mask) * 100 if 'total_pixels_in_mask' in locals() else (piksel_infeksi / total_piksel) * 100
        
        # Menampilkan hasil kalkulasi persentase dengan widget Metric Streamlit yang futuristik
        st.metric(label="Estimasi Luas Area Infeksi (Terhadap Total Luas Daun/Foto)", value=f"{persentase_luas_infeksi:.2f} %")
        
        # SISTEM PAKAR SEDERHANA: Memberikan rekomendasi penanganan otomatis berdasarkan tingkat keparahan luas area
        st.markdown("**📋 Rekomendasi Tindakan untuk Petani:**")
        if persentase_luas_infeksi < 10.0:
            st.info("🟢 **Kategori: Infeksi Ringan.** Penyakit masih terlokalisir. Cukup potong dan bakar daun yang bergejala ini agar tidak menular ke ranting tanaman yang lain.")
        elif persentase_luas_infeksi < 30.0:
            st.warning("🟡 **Kategori: Infeksi Sedang.** Penyakit mulai menyebar. Direkomendasikan melakukan penyemprotan cairan fungisida/pestisida organik secara berkala pada area sekitar daun.")
        else:
            st.error("🔴 **Kategori: Infeksi Parah!** Area kerusakan daun sudah sangat luas. Segera pisahkan/karantina tanaman ini dari tanaman sehat lainnya dan lakukan pemangkasan intensif untuk menghindari gagal panen massal.")
            
    else:
        # Jika tidak ada masker penyakit yang terdeteksi oleh AI
        st.balloons()
        st.success("🟢 **Tanaman Sehat!** Model AI tidak mendeteksi adanya gejala infeksi penyakit pada sampel daun apel ini.")