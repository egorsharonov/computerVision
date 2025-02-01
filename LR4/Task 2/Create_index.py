import os
from FAISS_index import create_and_save_embeddings

data_folder = "hse_faces_miem"
csv_file = os.path.join(data_folder, "staff_photo.csv")
create_and_save_embeddings(data_folder, csv_file)