import os

import streamlit as st
from PIL import Image
from find_face import find_closest_face


st.title("Поиск ближайшего лица")
uploaded_file = st.file_uploader("Загрузите изображение лица", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Сохранение загруженного изображения
    image_path = "uploaded_image.jpg"
    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    closest_name, closest_distance = find_closest_face(image_path)

    if closest_name:
        st.success(f"Наиболее похожий человек: **{closest_name}**")
        st.info(f"Расстояние: **{closest_distance:.4f}**")
    else:
        st.error("Лицо не найдено.")

    st.image(Image.open(image_path), caption="Загруженное изображение", use_container_width=True)