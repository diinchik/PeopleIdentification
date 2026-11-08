
import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path
import tempfile
import shutil
import os

import cv2
import numpy as np
from PIL import Image, ImageTk, ImageOps


# ==========================================
# 1. ГОЛОВНЕ ВІКНО ПРОГРАМИ
# ==========================================

root = tk.Tk()
root.title("Face Detection - Розпізнавання облич")
root.geometry("850x750")
root.configure(bg="#1e293b")
root.minsize(600, 550)


# ==========================================
# 2. ЗАВАНТАЖЕННЯ МОДЕЛІ YUNET
# ==========================================

MODEL_NAME = "face_detection_yunet_2023mar.onnx"


MODEL_PATH = Path(__file__).resolve().parent / MODEL_NAME


def load_detector():

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Не знайдено модель {MODEL_NAME}.\n"
            "Покладіть її в папку з lab5.py."
        )

    # Перевірка розміру моделі
    if MODEL_PATH.stat().st_size < 100000:
        raise RuntimeError(
            "Файл моделі занадто малий.\n"
            "Завантажте повний ONNX-файл YuNet."
        )

    # Створення тимчасової папки з ASCII-шляхом.
    # Це допомагає уникнути проблем OpenCV
    # з українськими літерами у Windows.
    temp_dir = Path(tempfile.gettempdir())

    # Використовуємо латинську назву файлу
    temp_model = temp_dir / MODEL_NAME

    # Копіювання моделі
    shutil.copyfile(MODEL_PATH, temp_model)

    # Завантаження YuNet
    detector = cv2.FaceDetectorYN.create(
        str(temp_model),
        "",
        (320, 320),
        0.65,
        0.3,
        5000
    )

    return detector


try:
    detector = load_detector()

except Exception as error:
    messagebox.showerror(
        "Помилка моделі",
        f"Не вдалося завантажити YuNet:\n{error}"
    )
    root.destroy()
    raise SystemExit(1)


# ==========================================
# 3. ФУНКЦІЯ ВИЯВЛЕННЯ ОБЛИЧ
# ==========================================

def detect_faces(image):

    height, width = image.shape[:2]

    # Обмеження розміру зображення
    max_dimension = 1280

    scale = min(
        1.0,
        max_dimension / max(width, height)
    )

    if scale < 1.0:

        new_width = max(1, round(width * scale))
        new_height = max(1, round(height * scale))

        processed = cv2.resize(
            image,
            (new_width, new_height),
            interpolation=cv2.INTER_AREA
        )

    else:
        processed = image

    h, w = processed.shape[:2]

    # Налаштування розміру зображення для моделі
    detector.setInputSize((w, h))

    # Детектування облич
    _, faces = detector.detect(processed)

    if faces is None:
        return []

    results = []

    for face in faces:

        x, y, fw, fh = face[:4]

        # Повернення координат до початкового розміру
        x = int(x / scale)
        y = int(y / scale)
        fw = int(fw / scale)
        fh = int(fh / scale)

        # Перевірка меж фотографії
        x1 = max(0, x)
        y1 = max(0, y)

        x2 = min(width, x + fw)
        y2 = min(height, y + fh)

        if x2 > x1 and y2 > y1:
            results.append(
                (x1, y1, x2 - x1, y2 - y1)
            )

    return results


# ==========================================
# 4. ВИБІР ФОТОГРАФІЇ
# ==========================================

def select_image():

    file_path = filedialog.askopenfilename(
        title="Виберіть фотографію",
        filetypes=[
            ("Зображення", "*.jpg *.jpeg *.png *.bmp"),
            ("Усі файли", "*.*")
        ]
    )

    if not file_path:
        return

    try:

        status_label.config(
            text="Виконується аналіз фотографії..."
        )
        root.update_idletasks()

        # Завантаження фотографії
        with Image.open(file_path) as source:
            pil_image = ImageOps.exif_transpose(
                source
            ).convert("RGB")

        # Перетворення у формат OpenCV
        image = cv2.cvtColor(
            np.array(pil_image),
            cv2.COLOR_RGB2BGR
        )

        # Пошук облич
        faces = detect_faces(image)

        # Малювання зелених прямокутників
        for (x, y, w, h) in faces:

            thickness = max(
                2,
                int(min(image.shape[:2]) / 350)
            )

            cv2.rectangle(
                image,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                thickness
            )

        # Перетворення зображення для Tkinter
        image_rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        result_image = Image.fromarray(image_rgb)

        # Зміна розміру зі збереженням пропорцій
        result_image.thumbnail(
            (740, 440),
            Image.Resampling.LANCZOS
        )

        photo = ImageTk.PhotoImage(result_image)

        # Відображення фотографії
        image_label.config(
            image=photo,
            text=""
        )

        image_label.image = photo

        # Кількість знайдених облич
        count = len(faces)

        count_label.config(
            text=f"Знайдено облич: {count}"
        )

        # Назва фотографії
        filename_label.config(
            text=Path(file_path).name
        )

        # Статус
        if count == 0:
            status_label.config(
                text="Обличчя не виявлено"
            )
        else:
            status_label.config(
                text="Аналіз фотографії завершено"
            )

        print(
            f"Фотографія: {Path(file_path).name}"
        )

        print(
            f"Знайдено облич: {count}"
        )

    except Exception as error:

        status_label.config(
            text="Помилка обробки фотографії"
        )

        messagebox.showerror(
            "Помилка",
            f"Не вдалося обробити фотографію:\n{error}"
        )


# ==========================================
# 5. ГРАФІЧНИЙ ІНТЕРФЕЙС
# ==========================================

# Заголовок
title_label = tk.Label(
    root,
    text="FACE DETECTION",
    font=("Arial", 25, "bold"),
    fg="#ffffff",
    bg="#1e293b"
)
title_label.pack(pady=(25, 5))


# Підзаголовок
subtitle_label = tk.Label(
    root,
    text="Автоматичне виявлення облич за допомогою OpenCV YuNet",
    font=("Arial", 12),
    fg="#94a3b8",
    bg="#1e293b"
)
subtitle_label.pack(pady=(0, 20))


# Область фотографії
image_frame = tk.Frame(
    root,
    bg="#334155",
    width=760,
    height=460
)
image_frame.pack(padx=20, pady=10)
image_frame.pack_propagate(False)


image_label = tk.Label(
    image_frame,
    text="Тут з'явиться ваша фотографія",
    font=("Arial", 14),
    fg="#cbd5e1",
    bg="#334155"
)
image_label.pack(expand=True)


# Назва файлу
filename_label = tk.Label(
    root,
    text="Фотографію не вибрано",
    font=("Arial", 10),
    fg="#94a3b8",
    bg="#1e293b"
)
filename_label.pack(pady=5)


# Кількість облич
count_label = tk.Label(
    root,
    text="Знайдено облич: 0",
    font=("Arial", 20, "bold"),
    fg="#4ade80",
    bg="#1e293b"
)
count_label.pack(pady=10)


# Кнопка вибору фотографії
select_button = tk.Button(
    root,
    text="Вибрати фотографію",
    command=select_image,
    font=("Arial", 13, "bold"),
    fg="white",
    bg="#2563eb",
    activebackground="#1d4ed8",
    activeforeground="white",
    relief="flat",
    padx=25,
    pady=12,
    cursor="hand2"
)
select_button.pack(pady=10)


# Статус
status_label = tk.Label(
    root,
    text="Очікування вибору фотографії...",
    font=("Arial", 10),
    fg="#94a3b8",
    bg="#1e293b"
)
status_label.pack(pady=5)


# ==========================================
# 6. ЗАПУСК ПРОГРАМИ
# ==========================================

root.mainloop()
