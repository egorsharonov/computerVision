import pandas as pd
from augment import augmentations, evaluate_precision
import os

# Пути к данным
reference_image = "reference.jpg"
output_folder = "augmented_dataset"

# Модели для тестирования
models = [
    "VGG-Face",
    "Facenet",
    "Facenet512",
    "OpenFace",
    "DeepFace",
    "DeepID",
    "ArcFace",
    "Dlib",
    "SFace",
    "GhostFaceNet"
]

# Сбор результатов
results = []

for model in models:
    row = {"Model": model}

    # Проходим по всем аугментированным датасетам
    for aug_name in augmentations.keys():
        aug_folder = os.path.join(output_folder, aug_name.replace(" ", "_").lower())
        dataset_paths = [os.path.join(aug_folder, f) for f in os.listdir(aug_folder)]

        # Оцениваем Precision
        precision = evaluate_precision(reference_image, dataset_paths, aug_name, model_name=model)
        row[aug_name] = precision

    results.append(row)

# Создание таблицы
df_results = pd.DataFrame(results)
print(df_results)

# Сохранение в CSV
df_results.to_csv("precision_results2.csv", index=False)