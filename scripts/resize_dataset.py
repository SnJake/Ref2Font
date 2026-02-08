import os
from PIL import Image
from tqdm import tqdm

# Настройки
TARGET_SIZE = (1024, 1024)
BASE_DIR = "E:\Fonts"
SUBDIRS = ["targets", "controls"] # Папки для обработки

def resize_images():
    for subdir in SUBDIRS:
        input_path = os.path.join(BASE_DIR, subdir)
        # Создаем новую папку, чтобы не испортить оригиналы (опционально)
        output_path = os.path.join(BASE_DIR, f"{subdir}_1024")
        
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        
        print(f"Обработка папки: {subdir}...")
        
        files = [f for f in os.listdir(input_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
        
        for filename in tqdm(files):
            img_path = os.path.join(input_path, filename)
            try:
                with Image.open(img_path) as img:
                    # Ресайз с использованием высококачественного фильтра LANCZOS
                    resized_img = img.resize(TARGET_SIZE, Image.Resampling.LANCZOS)
                    
                    # Сохраняем в WebP Lossless для сохранения идеальных краев
                    save_path = os.path.join(output_path, filename)
                    resized_img.save(save_path, "WEBP", lossless=True)
            except Exception as e:
                print(f"Ошибка в файле {filename}: {e}")

    print("\nГотово! Новые изображения лежат в папках targets_1024 и controls_1024")

if __name__ == "__main__":
    resize_images()