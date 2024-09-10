import cv2
import pickle
import numpy as np
import os
from tqdm import tqdm

dataset_path = 'Kaggle/Dataset/Celebrity Face Recognition'

faceCascade = cv2.CascadeClassifier('Haarcascade/haarcascade_frontalface_default.xml')

color = {"Blue": (255, 0, 0), "Red": (0, 0, 255), "Green": (0, 255, 0), "White": (255, 255, 255)}

faces_data = []
names = []

def process_images_from_folder(folder_name, label):
    global faces_data, names
    folder_path = os.path.join(dataset_path, folder_name)
    for filename in tqdm(os.listdir(folder_path), desc=f"Processing {folder_name}"):
        if filename.endswith(("jpg", "png", "jpeg")):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = faceCascade.detectMultiScale(
                gray_img, 
                scaleFactor=1.1, 
                minNeighbors=5, 
                minSize=(30, 30), 
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            for (x, y, w, h) in faces:
                cropped_img = img[y:y+h, x:x+w, :]
                resized_img = cv2.resize(cropped_img, (50, 50))
                for _ in range(100):
                    faces_data.append(resized_img)
                    names.append(label)

for folder_name in os.listdir(dataset_path):
    folder_path = os.path.join(dataset_path, folder_name)
    if os.path.isdir(folder_path):
        print(f"Memproses gambar untuk wajah: {folder_name}")
        process_images_from_folder(folder_name, folder_name)

faces_data = np.asarray(faces_data)
faces_data = faces_data.reshape(faces_data.shape[0], -1)

if not os.path.exists('data'):
    os.makedirs('data')

with open('data/faces_data.pkl', 'wb') as f:
    pickle.dump(faces_data, f)

with open('data/names_data.pkl', 'wb') as f:
    pickle.dump(names, f)

print("Data wajah dan nama telah disimpan ke dalam database.")