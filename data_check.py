import pickle

with open('Data/faces_data.pkl', 'rb') as file_faces, open('Data/names_data.pkl', 'rb') as file_names:
    faces_data = pickle.load(file_faces)
    names_data = pickle.load(file_names)

print(faces_data)
print(names_data)