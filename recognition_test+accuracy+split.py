from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import cv2
import pickle

video_capture = cv2.VideoCapture(0)
faceCascade = cv2.CascadeClassifier('Haarcascade/haarcascade_frontalface_default.xml')
color = {"Blue": (255, 0, 0), "Red": (0, 0, 255), "Green": (0, 255, 0), "White": (255, 255, 255)}

with open('Data/names_data.pkl', 'rb') as w:
    LABELS = pickle.load(w)
with open('Data/faces_data.pkl', 'rb') as f:
    FACES = pickle.load(f)

print('Shape of Faces matrix --> ', FACES.shape)

# Split the data into training and test sets
FACES_train, FACES_test, LABELS_train, LABELS_test = train_test_split(FACES, LABELS, test_size=0.2, random_state=42)

# Initialize and train the KNN model
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(FACES_train, LABELS_train)

# Evaluate the model on the test set
accuracy = knn.score(FACES_test, LABELS_test)
print(f'Accuracy on test set: {accuracy:.2f}')

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
        cropped_img = img[y:y + h, x:x + w, :]
        resized_img = cv2.resize(cropped_img, (50, 50)).flatten().reshape(1, -1)

        output = knn.predict(resized_img)
        prob = knn.predict_proba(resized_img).max() * 100

        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 1)
        cv2.rectangle(img, (x, y), (x + w, y + h), (50, 50, 255), 2)
        cv2.rectangle(img, (x, y - 40), (x + w, y), (50, 50, 255), -1)
        text = f"{str(output[0])} ({prob:.2f}%)"
        cv2.putText(img, text, (x, y - 15), cv2.FONT_HERSHEY_COMPLEX, 1, color['White'], 1)
        cv2.rectangle(img, (x, y), (x + w, y + h), (50, 50, 255), 1)

    cv2.imshow("Frame", img)

    key = cv2.waitKey(1)
    if key == ord('q') or key == ord('Q'):
        break

video_capture.release()
cv2.destroyAllWindows()