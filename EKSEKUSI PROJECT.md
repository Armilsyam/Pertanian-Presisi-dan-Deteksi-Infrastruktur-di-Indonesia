🔑 Langkah Awal Eksekusi Project
1. Domain Fokus
Pilih sektor utama:

🌱 Pertanian Presisi → deteksi penyakit tanaman.

🏗️ Infrastruktur → deteksi kerusakan jalan/jembatan.
👉 Saran: mulai dari pertanian, karena data lebih mudah dikumpulkan (foto daun/tanaman).

2. Pengumpulan Data
Dataset Awal: kumpulkan ±500–1000 gambar daun sehat vs sakit.

Gunakan kamera smartphone atau drone sederhana.

Tambahkan sensor IoT (kelembaban, suhu) untuk multimodal data.

Buat folder terstruktur:

/images/healthy

/images/diseased

/sensor_data/

3. Pengembangan Model
Framework: PyTorch / TensorFlow.

Arsitektur: Vision Transformer (ViT) → lakukan pruning & quantization.

Integrasi multimodal: gabungkan citra + data sensor (early fusion).

Tambahkan modul Explainable AI (Grad-CAM, saliency maps) untuk visualisasi hasil.

4. Deployment
Target perangkat: Raspberry Pi 4 atau NVIDIA Jetson Nano.

Optimasi model agar berjalan real-time dengan konsumsi energi rendah.

Buat dashboard sederhana (misalnya dengan Streamlit) untuk menampilkan hasil deteksi + heatmap penjelasan.

5. Evaluasi
Uji akurasi model dengan dataset validasi.

Uji kecepatan inference di perangkat edge.

Uji interpretabilitas (apakah visualisasi mudah dipahami petani/teknisi).

6. Publikasi & Presentasi
Susun laporan hasil eksperimen.

Draft artikel untuk jurnal (IEEE/Scopus).

Presentasi di kelas/forum riset UNPAM.

📌 Langkah paling konkret yang bisa dilakukan minggu ini:

Tentukan domain fokus (pertanian atau infrastruktur).

Mulai kumpulkan dataset kecil (foto tanaman/infrastruktur).

Setup environment coding (Python + PyTorch/TensorFlow).
