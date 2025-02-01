import cv2
import numpy as np
from deepface import DeepFace
from tqdm import tqdm


def rotate_image(image, angle):
    """
    Поворачивает изображение на заданный угол.

    :param image: Исходное изображение (numpy array)
    :param angle: Угол поворота (в градусах)
    :return: Повернутое изображение
    """
    height, width = image.shape[:2]
    center = (int(width / 2), int(height / 2))
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)
    cos = np.abs(rotation_matrix[0, 0])
    sin = np.abs(rotation_matrix[0, 1])
    new_width = int((height * abs(sin)) + (width * abs(cos)))
    new_height = int((height * abs(cos)) + (width * abs(sin)))
    rotated_image = cv2.warpAffine(image, rotation_matrix, (new_width, new_height))

    return rotated_image


def add_noise(image, mean=0, sigma=25):
    """Добавляет гауссовский шум к изображению."""
    noise = np.random.normal(mean, sigma, image.shape).astype(np.uint8)
    noisy_image = cv2.add(image, noise)
    return noisy_image


def adjust_brightness(image, factor):
    """Изменяет яркость изображения."""
    adjusted = cv2.convertScaleAbs(image, alpha=factor, beta=0)
    return adjusted


def apply_blur(image, kernel_size=(5, 5)):
    """Применяет размытие к изображению."""
    blurred = cv2.GaussianBlur(image, kernel_size, 0)
    return blurred


def evaluate_precision(reference_image, dataset_images, aug, model_name="Facenet"):
    """
    Оценивает Precision для датасета изображений.

    :param reference_image: Путь к эталонному изображению
    :param dataset_images: Список путей к изображениям датасета
    :param model_name: Название модели для сравнения
    :return: Значение Precision
    """
    true_positives = 0
    total_predictions = len(dataset_images)

    for image_path in tqdm(dataset_images, desc=f"Обработка изображений датасета {aug} моделью {model_name} ", unit="img"):
        try:
            result = DeepFace.verify(
                img1_path=reference_image,
                img2_path=image_path,
                model_name=model_name,
                enforce_detection=False
            )
            if result["verified"]:
                true_positives += 1
        except Exception as e:
            print(f"Ошибка при обработке {image_path}: {e}")

    precision = true_positives / total_predictions if total_predictions > 0 else 0
    return precision

# Аугментации
augmentations = {
    "Original": lambda img: img,
    "Rotate 45": lambda img: rotate_image(img, 45),
    "Rotate 90": lambda img: rotate_image(img, 90),
    "Gaussian Noise": lambda img: add_noise(img),
    "Brightness +50%": lambda img: adjust_brightness(img, 1.5),
    "Brightness -50%": lambda img: adjust_brightness(img, 0.5),
    "Blur (5x5)": lambda img: apply_blur(img, (5, 5))
}