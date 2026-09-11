# Foto_Kita_Blur

Aplikasi kamera real-time berbasis Python yang otomatis mem-blur layar dan memunculkan hati-hati pink beterbangan saat kamu menunjukkan gesture ✌️ (peace sign) ke webcam — terinspirasi dari tren **"foto kita blur"** yang viral di TikTok.

## ✨ Fitur

- **Deteksi gesture tangan real-time** pakai MediaPipe Hand Landmarker (Tasks API terbaru)
- **Efek blur otomatis** ke seluruh layar saat gesture ✌️ terdeteksi
- **Hati-hati pink beterbangan** yang muncul acak, naik ke atas, dan memudar (particle effect)
- **Visualisasi skeleton tangan** — titik & garis sendi jari yang keliatan live saat tangan di-track
- **Notifikasi toast** "Blur ON!" / "Blur OFF" yang muncul sesaat tiap status berubah
- **Label instruksi** yang selalu tampil di bawah layar
- **Musik latar** yang otomatis diputar & loop selama aplikasi berjalan (opsional)
- **Peningkatan kualitas kamera** — resolusi lebih tinggi, kontras & sharpening otomatis

## 🛠️ Requirements

- Python 3.10 – 3.13 direkomendasikan (Python 3.14 masih ada kendala kompatibilitas paket `pygame`, pakai `pygame-ce` sebagai gantinya — lihat bagian Instalasi)
- Webcam
- Koneksi internet (buat download model AI di run pertama)

## 📦 Instalasi

```bash
pip install opencv-python mediapipe numpy pygame-ce
```

> **Catatan:** kalau kamu pakai Python 3.14 dan `pip install pygame` gagal, itu wajar — paket `pygame` resmi belum ada wheel buat Python 3.14. Gunakan `pygame-ce` (Community Edition) sebagai gantinya, kodenya tetap sama persis karena `import pygame` tetap dipakai.

## ▶️ Cara Pakai

1. Clone repo ini
2. (Opsional) taruh file musik (`.mp3`) di folder yang sama, lalu sesuaikan nama file di variabel `MUSIC_PATH` pada `foto_kita_blur.py`
3. Jalankan:
   ```bash
   python foto_kita_blur.py
   ```
4. Model AI (`hand_landmarker.task`, ±7MB) akan otomatis terunduh di run pertama
5. Tunjukkan gesture ✌️ ke webcam untuk memicu efek blur + love
6. Tekan `q` untuk keluar

## ⚙️ Kustomisasi

Beberapa hal yang gampang diubah langsung di kode:

| Ingin ubah... | Cari variabel/baris |
|---|---|
| Warna hati | `PINK_FILL`, `PINK_OUTLINE` |
| Ukuran hati utama | `int(45 * pulse)` |
| Jumlah/kepadatan hati kecil | `spawn_attempts`, peluang `random.random() >= 0.5` |
| Teks instruksi bawah | string `"Angkat 2 jari untuk blur + love"` |
| Sensitivitas deteksi tangan | `min_hand_detection_confidence`, `min_tracking_confidence` |
| Kekuatan sharpening/kontras | `alpha`, `beta` di `enhance_frame()` |

## 🙏 Credit

Terinspirasi dari tren **"Foto Kita Blur"** yang viral di TikTok. Dibangun dengan [OpenCV](https://opencv.org/) dan [MediaPipe](https://ai.google.dev/edge/mediapipe).

## 📄 License

MIT — bebas dipakai, dimodifikasi, dan dibagikan.
