import cv2
import os
import numpy as np

IMG_SIZE = 128

def load_images(dataset_path):
    data = []
    labels = []

    for category in ["real", "fake"]:
        path = os.path.join(dataset_path, category)

        for img in os.listdir(path):
            img_path = os.path.join(path, img)

            image = cv2.imread(img_path)
            image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

            data.append(image)

            if category == "real":
                labels.append(0)
            else:
                labels.append(1)

    return np.array(data), np.array(labels)