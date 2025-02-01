import cv2
import numpy as np

model_path = "face_detection_yunet_2023mar.onnx"
detector = cv2.FaceDetectorYN_create(model_path, "", (0, 0))

def calculate_face_distance(face1, face2):
    return np.linalg.norm(np.array(face1) - np.array(face2))


my_face_keypoints = None


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Не удалось получить кадр")
        break

    # Получение размеров кадра
    height, width = frame.shape[:2]
    detector.setInputSize((width, height))

    # Обнаружение лиц
    faces = detector.detect(frame)

    if faces[1] is not None:
        for face in faces[1]:
            # Координаты прямоугольника вокруг лица
            box = list(map(int, face[:4]))
            x, y, w, h = box

            # Ключевые точки лица
            keypoints = list(map(int, face[4:14]))
            keypoints = [(keypoints[i], keypoints[i + 1]) for i in range(0, len(keypoints), 2)]

            # Определяем, является ли лицо вашим
            if my_face_keypoints is None:
                # Если моё лицо ещё не сохранено, сохраняем его
                my_face_keypoints = keypoints
                color = (0, 255, 0)  # Зелёный
            else:
                # Сравниваем расстояние между ключевыми точками
                distance = calculate_face_distance(my_face_keypoints, keypoints)
                if distance < 200:  # Пороговое значение
                    color = (0, 255, 0)  # Зелёный
                else:
                    color = (0, 0, 255)  # Красный

            # Рисуем прямоугольник вокруг лица
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            # Рисуем ключевые точки лица
            for point in keypoints:
                cv2.circle(frame, point, 2, (255, 0, 0), -1)

    # Отображение результата
    cv2.imshow("Face Detection", frame)

    # Выход по нажатию клавиши 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождение ресурсов
cap.release()
cv2.destroyAllWindows()