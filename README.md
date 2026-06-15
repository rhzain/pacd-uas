# Homography Perspective Tool

Aplikasi sederhana untuk projek UAS Pengolahan Analisis Citra Digital:

- koreksi perspektif area miring menjadi bidang datar;
- proyeksi gambar overlay ke bidang yang dipilih;
- seleksi 4 titik manual;
- auto detection quadrilateral berbasis Canny edge dan contour detection;
- preview polygon, before-after, homography matrix, dan download output.

## Cara Menjalankan

Pastikan Python sudah terpasang. Dari folder `program`, jalankan:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

Jika command `python` belum tersedia di Windows, install Python dari https://www.python.org/downloads/ lalu centang opsi `Add python.exe to PATH`.

## Alur Pemakaian

1. Upload gambar utama.
2. Pilih mode area:
   - `Manual Selection`: klik 4 titik pada gambar atau isi koordinat manual.
   - `Auto Detection`: aplikasi mencari quadrilateral terbesar dari hasil edge detection.
3. Pastikan titik berurutan `top-left`, `top-right`, `bottom-right`, `bottom-left`.
4. Pilih aksi:
   - `Perspective Correction` untuk meluruskan area.
   - `Image Projection` untuk menempelkan gambar overlay ke area.
5. Download hasil.

## Konsep Utama

Kedua fitur menggunakan homography dari empat pasang titik korespondensi:

```text
source points + destination points
-> cv2.getPerspectiveTransform
-> cv2.warpPerspective
-> output image
```

Pada correction, area miring menjadi rectangle datar. Pada projection, rectangle overlay dipetakan ke area miring pada background.

## Tips Auto Detection

Jika area terdeteksi tetapi kurang akurat:

- aktifkan `Tampilkan edge detection` untuk melihat tepi yang dibaca sistem;
- turunkan `Canny low/high` jika tepi billboard tidak muncul;
- naikkan `Canny low/high` jika terlalu banyak garis dari background atau tulisan;
- naikkan `Close iterations` jika garis tepi putus-putus;
- biarkan `Dilate iterations` di `0` untuk hasil titik yang lebih presisi;
- naikkan `Min area (%)` jika sistem memilih objek kecil yang salah.

Jika foto terlalu ramai atau tepi billboard tidak jelas, gunakan auto detection sebagai titik awal lalu koreksi koordinat manual.
