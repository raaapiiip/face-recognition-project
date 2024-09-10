import cv2
import pickle
import numpy as np
import os

video_capture = cv2.VideoCapture(0)
faceCascade = cv2.CascadeClassifier('Haarcascade/haarcascade_frontalface_default.xml')
color = {"Blue":(255, 0, 0), "Red": (0, 0, 255), "Green": (0, 255, 0), "White":(255, 255, 255)}

faces_data = []
count = 0

name = input("Masukkan nama anda: ")

while True:
    boolean, img = video_capture.read()
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
        resized_img = cv2.resize(cropped_img, (50,50))
        
        if len(faces_data) <= 50 and count%10 == 0:
            faces_data.append(resized_img)
        
        count = count+1
        cv2.putText(img, str(len(faces_data)), (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, color['Blue'], 1)
        cv2.rectangle(img, (x,y), (x+w, y+h), color['Blue'], 1)
    
    cv2.imshow("Frame", img)
    
    key = cv2.waitKey(1)
    if key == ord('q') or key == ord('Q') or len(faces_data) == 50:
        break

video_capture.release()
cv2.destroyAllWindows()

faces_data = np.asarray(faces_data)
faces_data = faces_data.reshape(50, -1)

if 'names_data.pkl' not in os.listdir('data/'):
    names = [name]*50
    with open('Data/names_data.pkl', 'wb') as f:
        pickle.dump(names, f)
else:
    with open('Data/names_data.pkl', 'rb') as f:
        names = pickle.load(f)
    names = names+[name]*50
    with open('Data/names_data.pkl', 'wb') as f:
        pickle.dump(names, f)

if 'faces_data.pkl' not in os.listdir('data/'):
    with open('Data/faces_data.pkl', 'wb') as f:
        pickle.dump(faces_data, f)
else:
    with open('Data/faces_data.pkl', 'rb') as f:
        faces = pickle.load(f)
    faces = np.append(faces, faces_data, axis=0)
    with open('Data/faces_data.pkl', 'wb') as f:
        pickle.dump(faces, f)