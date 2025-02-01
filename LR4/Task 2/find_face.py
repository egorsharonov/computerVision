from FAISS_index import extract_embedding, load_embeddings_and_index


def find_closest_face(query_image_path, embeddings_folder="embeddings", model_name="Facenet", top_k=1):
    """
    Поиск ближайшего лица в FAISS-индексе.

    :param query_image_path: Путь к изображению запроса
    :param embeddings_folder: Папка с сохранёнными данными
    :param model_name: Название модели для извлечения эмбеддингов
    :param top_k: Количество ближайших соседей для поиска
    :return: Имя человека и расстояние до него
    """
    _, index, metadata = load_embeddings_and_index(embeddings_folder)

    query_embedding = extract_embedding(query_image_path, model_name=model_name)
    if query_embedding is None:
        return None, None

    # Поиск ближайших соседей
    distances, indices = index.search(query_embedding.reshape(1, -1), top_k)

    # Получение метаданных для найденных индексов
    closest_match = metadata[indices[0][0]]
    closest_name = closest_match[1]
    closest_distance = distances[0][0]

    return closest_name, closest_distance