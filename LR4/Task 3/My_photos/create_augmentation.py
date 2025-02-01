from augment import augmentations
import os
import cv2


# Пути к данным
dataset_folder = "dataset"
output_folder = "augmented_dataset"
os.makedirs(output_folder, exist_ok=True)
# Применение аугментаций ко всему датасету
for aug_name, aug_func in augmentations.items():
    aug_folder = os.path.join(output_folder, aug_name.replace(" ", "_").lower())
    os.makedirs(aug_folder, exist_ok=True)

    for image_file in os.listdir(dataset_folder):
        image_path = os.path.join(dataset_folder, image_file)
        image = cv2.imread(image_path)
        if image is None:
            print(f"Не удалось загрузить изображение: {image_path}")
            continue
        augmented_image = aug_func(image)
        output_path = os.path.join(aug_folder, image_file)
        saved = cv2.imwrite(output_path, augmented_image)
        if not saved:
            print("Не удалось сохранить изображение по пути:", output_path)

print("Аугментации успешно применены.")