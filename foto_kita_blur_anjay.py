import os
import time
import random
import urllib.request

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import pygame

# Path file musik latar - ganti sesuai nama file kamu (mp3/ogg/wav)
MUSIC_PATH = "musik.mp3"

# File model buat deteksi tangan (Tasks API baru dari MediaPipe >= 1.0,
# menggantikan mp.solutions.hands yang sudah dihapus)
MODEL_PATH = "hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)

# Download model otomatis kalau belum ada di folder ini
if not os.path.exists(MODEL_PATH):
    print("Mengunduh model hand_landmarker.task (sekali saja)...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Model selesai diunduh.")

# Inisialisasi HandLandmarker (mode VIDEO: cocok buat frame demi frame dari webcam)
hand_landmarker = mp_vision.HandLandmarker.create_from_options(
    mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    )
)

# Inisialisasi & mainkan musik latar (loop terus selama app jalan)
pygame.mixer.init()
if os.path.exists(MUSIC_PATH):
    pygame.mixer.music.load(MUSIC_PATH)
    pygame.mixer.music.play(loops=-1)  # -1 = ulang terus tanpa henti
else:
    print(f"[!] File musik '{MUSIC_PATH}' tidak ditemukan, lanjut tanpa musik.")

# Inisialisasi kamera
cap = cv2.VideoCapture(0)
start_time = time.time()


def is_peace_sign(landmarks):
    """
    Cek apakah landmark tangan membentuk gesture peace sign (✌️).
    `landmarks` = list berisi 21 titik (NormalizedLandmark), masing2 punya .x/.y/.z

    Index landmark MediaPipe Hands (index tetap sama seperti sebelumnya):
      Index  : MCP=5,  PIP=6,  DIP=7,  TIP=8
      Middle : MCP=9,  PIP=10, DIP=11, TIP=12
      Ring   : MCP=13, PIP=14, DIP=15, TIP=16
      Pinky  : MCP=17, PIP=18, DIP=19, TIP=20
    Ingat: sumbu Y di gambar itu terbalik -> makin ke atas nilainya makin kecil.
    Jadi "jari berdiri/naik" berarti TIP.y < PIP.y
    """
    index_up = landmarks[8].y < landmarks[6].y
    middle_up = landmarks[12].y < landmarks[10].y
    ring_down = landmarks[16].y > landmarks[14].y
    pinky_down = landmarks[20].y > landmarks[18].y

    return index_up and middle_up and ring_down and pinky_down


PINK_FILL = (180, 105, 255)      # warna pink (format BGR)
PINK_OUTLINE = (130, 70, 220)    # pink lebih gelap buat garis tepi


def draw_heart(img, center, size, color=PINK_FILL, outline=PINK_OUTLINE, alpha=1.0):
    """
    Gambar bentuk hati (love) di atas `img`.
    center: (x, y) titik tengah hati
    size  : ukuran kira-kira lebar/tinggi hati dalam pixel
    alpha : 1.0 = solid penuh, makin kecil makin transparan
    """
    t = np.linspace(0, 2 * np.pi, 100)
    x = 16 * np.sin(t) ** 3
    y = -(13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t))

    scale = size / 32.0  # kurva dasar lebarnya sekitar 32 unit
    pts = np.stack(
        [x * scale + center[0], y * scale + center[1]], axis=1
    ).astype(np.int32)

    if alpha >= 0.999:
        cv2.fillPoly(img, [pts], color, lineType=cv2.LINE_AA)
        cv2.polylines(img, [pts], isClosed=True, color=outline,
                      thickness=1, lineType=cv2.LINE_AA)
    else:
        # Gambar di layer terpisah dulu, baru dicampur transparan ke frame asli
        overlay = img.copy()
        cv2.fillPoly(overlay, [pts], color, lineType=cv2.LINE_AA)
        cv2.polylines(overlay, [pts], isClosed=True, color=outline,
                      thickness=1, lineType=cv2.LINE_AA)
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, dst=img)


# Daftar koneksi antar titik tangan (buat gambar skeleton), diambil dari API baru
HAND_CONNECTIONS = [
    (c.start, c.end) for c in mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS
]


def draw_hand_skeleton(img, landmarks, color=PINK_FILL, dot_color=(255, 255, 255)):
    """
    Gambar titik-titik sendi tangan + garis penghubungnya (skeleton),
    biar keliatan tangannya lagi di-track live.
    `landmarks`: list 21 NormalizedLandmark (koordinat 0.0-1.0), dikonversi ke pixel di sini.
    """
    h, w = img.shape[:2]
    points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

    for start_idx, end_idx in HAND_CONNECTIONS:
        cv2.line(img, points[start_idx], points[end_idx], color, 2, cv2.LINE_AA)

    for x, y in points:
        cv2.circle(img, (x, y), 4, dot_color, -1, cv2.LINE_AA)
        cv2.circle(img, (x, y), 4, color, 1, cv2.LINE_AA)


def draw_pill(img, center, width, height, color, alpha=1.0):
    """Gambar kotak pil (stadium shape) - rectangle dengan ujung bulat penuh."""
    x, y = center
    half_w, half_h = width // 2, height // 2
    radius = half_h

    def _draw(target):
        cv2.rectangle(target, (x - half_w + radius, y - half_h),
                      (x + half_w - radius, y + half_h), color, -1, cv2.LINE_AA)
        cv2.circle(target, (x - half_w + radius, y), radius, color, -1, cv2.LINE_AA)
        cv2.circle(target, (x + half_w - radius, y), radius, color, -1, cv2.LINE_AA)

    if alpha >= 0.999:
        _draw(img)
    else:
        overlay = img.copy()
        _draw(overlay)
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, dst=img)


def put_text_centered(img, text, center, font_scale=0.55, color=(255, 255, 255),
                       thickness=1, font=cv2.FONT_HERSHEY_DUPLEX):
    """Tulis teks yang otomatis center secara horizontal & vertikal di titik `center`."""
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    x = center[0] - tw // 2
    y = center[1] + th // 2
    cv2.putText(img, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)


# Daftar hati kecil yang lagi "terbang" ke atas (efek partikel)
floating_hearts = []

# State buat notifikasi toast "Blur ON/OFF" & instruksi bawah
was_peace_detected = False
toast_message = None
toast_start_time = 0.0
TOAST_DURATION = 1.5  # detik


while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Mirror biar kayak cermin (lebih enak buat selfie-cam)
    frame = cv2.flip(frame, 1)

    # Konversi ke RGB karena MediaPipe butuh RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # Timestamp wajib naik terus di mode VIDEO
    timestamp_ms = int((time.time() - start_time) * 1000)
    result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)

    peace_detected = False

    if result.hand_landmarks:
        for hand_landmarks in result.hand_landmarks:
            if is_peace_sign(hand_landmarks):
                peace_detected = True

    if peace_detected:
        # Terapkan efek blur (Gaussian blur) ke seluruh frame
        output_frame = cv2.GaussianBlur(frame, (55, 55), 0)

        cx = output_frame.shape[1] // 2
        base_y = output_frame.shape[0] - 90

        # Hati utama di tengah, ukurannya dikecilin & pink, "berdenyut" pelan
        pulse = 1.0 + 0.1 * np.sin(time.time() * 4)
        draw_heart(output_frame, (cx, base_y), int(45 * pulse))

        # Sesekali munculin hati kecil baru, tapi dicek dulu jaraknya
        # ke hati-hati lain biar nggak numpuk/tabrakan
        # (nyoba beberapa kali per frame biar lebih rame kalau ternyata kepentok jarak)
        spawn_attempts = 3
        for _ in range(spawn_attempts):
            if random.random() >= 0.5:
                continue
            candidate_x = cx + random.randint(-160, 160)
            candidate_y = base_y + random.randint(-20, 20)
            min_distance = 32

            too_close = any(
                ((p["x"] - candidate_x) ** 2 + (p["y"] - candidate_y) ** 2) ** 0.5 < min_distance
                for p in floating_hearts
            )

            if not too_close:
                floating_hearts.append({
                    "x": candidate_x,
                    "y": candidate_y,
                    "size": random.randint(10, 22),
                    "age": 0,
                    "max_age": random.randint(45, 75),
                })

        # Update posisi & gambar tiap hati kecil, buang yang udah "mati"
        still_alive = []
        for p in floating_hearts:
            p["age"] += 1
            progress = p["age"] / p["max_age"]
            if progress >= 1.0:
                continue
            y_pos = p["y"] - int(progress * 200)      # makin lama makin naik
            x_wobble = p["x"] + int(8 * np.sin(progress * 6 + p["x"]))  # goyang dikit biar natural
            alpha = 1.0 - progress                     # makin naik makin transparan
            draw_heart(output_frame, (x_wobble, y_pos), p["size"], alpha=alpha)
            still_alive.append(p)
        floating_hearts = still_alive
    else:
        output_frame = frame
        floating_hearts = [] 

    if result.hand_landmarks:
        for hand_landmarks in result.hand_landmarks:
            draw_hand_skeleton(output_frame, hand_landmarks)

    frame_h, frame_w = output_frame.shape[:2]

    # Toast notifikasi "Blur ON/OFF" di atas, cuma tampil sesaat & fade out
    if toast_message is not None:
        elapsed = time.time() - toast_start_time
        if elapsed < TOAST_DURATION:
            toast_alpha = 1.0 - (elapsed / TOAST_DURATION)
            draw_pill(output_frame, (frame_w // 2, 50), 160, 44, PINK_FILL, alpha=toast_alpha * 0.9)
            put_text_centered(output_frame, toast_message, (frame_w // 2, 50),
                               font_scale=0.65, thickness=2)
        else:
            toast_message = None

    cv2.imshow('Foto Kita Blur', output_frame)

    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

pygame.mixer.music.stop()
pygame.mixer.quit()
hand_landmarker.close()
cap.release()
cv2.destroyAllWindows()