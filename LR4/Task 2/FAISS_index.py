import os
import pandas as pd
import numpy as np
import faiss
from deepface import DeepFace
from tqdm import tqdm


def extract_embedding(image_path, model_name="Facenet"):
    try:
        result = DeepFace.represent(img_path=image_path, model_name=model_name, enforce_detection=False)
        embedding = result[0]['embedding']
        return np.array(embedding).astype("float32")
    except Exception as e:
        print(f"Ошибка при извлечении эмбеддинга из {image_path}: {e}")
        return None


def create_and_save_embeddings(data_folder, csv_file, output_folder="embeddings"):
    model_name = "Facenet"
    embedding_size = 128

    os.makedirs(output_folder, exist_ok=True)

    df = pd.read_csv(csv_file)

    embeddings_list = []
    metadata = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing images"):
        image_path = os.path.join(data_folder, row["filename"])
        if not os.path.exists(image_path):
            print(f"Файл не найден: {image_path}")
            continue

        embedding = extract_embedding(image_path, model_name=model_name)
        if embedding is None:
            continue

        embeddings_list.append(embedding)
        metadata.append((row["id"], row["name"]))

    embeddings_array = np.vstack(embeddings_list)

    # Создание FAISS-индекса
    index = faiss.IndexFlatL2(embedding_size)
    index.add(embeddings_array)

    # Сохранение
    np.save(os.path.join(output_folder, "embeddings.npy"), embeddings_array)
    faiss.write_index(index, os.path.join(output_folder, "faiss_index.index"))
    pd.DataFrame(metadata, columns=["id", "name"]).to_csv(os.path.join(output_folder, "metadata.csv"), index=False)

    print(f"Эмбеддинги, индекс и метаданные успешно сохранены в папку '{output_folder}'.")


def load_embeddings_and_index(embeddings_folder="embeddings"):
    """
    Загрузка эмбеддингов, FAISS-индекса и метаданных из файлов.

    :param embeddings_folder: Папка с сохранёнными данными
    :return: embeddings_array, index, metadata
    """
    embeddings_array = np.load(os.path.join(embeddings_folder, "embeddings.npy"))
    index = faiss.read_index(os.path.join(embeddings_folder, "faiss_index.index"))
    metadata_df = pd.read_csv(os.path.join(embeddings_folder, "metadata.csv"))
    metadata = list(zip(metadata_df["id"], metadata_df["name"]))

    return embeddings_array, index, metadata