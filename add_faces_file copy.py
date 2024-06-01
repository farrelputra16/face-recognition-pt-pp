import streamlit as st
import cv2
import pickle
import numpy as np
import os

# Inisialisasi classifier wajah
facedetect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

# Buffer untuk menyimpan data wajah dan nama
faces_data = []
names = []

# Inisialisasi atau muat ID counter
if 'data/id_counter.pkl' in os.listdir('data/'):
    with open('data/id_counter.pkl', 'rb') as f:
        id_counter = pickle.load(f)
else:
    id_counter = 1

# Tambahkan tombol untuk mengunggah gambar
uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Baca file gambar yang diunggah
    image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
    
    # Deteksi wajah dalam gambar
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)
    
    # Masukkan nama pengguna
    name = st.text_input("Enter Your Name")

    for _ in range(15):  # Loop untuk mengunggah dan melatih setiap gambar 15 kali
        for (x, y, w, h) in faces:
            # Pemotongan wajah dari gambar
            crop_img = image[y:y+h, x:x+w, :]
            resized_img = cv2.resize(crop_img, (50, 50))
            
            # Menambahkan wajah dan nama ke buffer
            faces_data.append(resized_img.flatten())
            names.append((id_counter, name))  # Gunakan ID dan nama

            # Tampilkan kotak di sekitar wajah
            cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Tampilkan gambar dengan kotak di sekitar wajah
    st.image(image, channels="BGR")

    # Tombol "Train Model" untuk melatih model
    if len(faces_data) > 0:
        if st.button("Train Model"):
            # Ubah buffer wajah dan nama ke dalam format yang sesuai untuk penyimpanan
            faces_data = np.array(faces_data)
            names = np.array(names)

            # Simpan dataset wajah
            if os.path.exists('data/faces_data.pkl'):
                with open('data/faces_data.pkl', 'rb') as f:
                    existing_faces_data = pickle.load(f)
                faces_data = np.concatenate((existing_faces_data, faces_data), axis=0)
            
            with open('data/faces_data.pkl', 'wb') as f:
                pickle.dump(faces_data, f)

            # Simpan dataset nama
            if os.path.exists('data/names.pkl'):
                with open('data/names.pkl', 'rb') as f:
                    existing_names = pickle.load(f)
                names = np.concatenate((existing_names, names), axis=0)
            
            with open('data/names.pkl', 'wb') as f:
                pickle.dump(names, f)

            # Increment ID counter
            id_counter += 1
            with open('data/id_counter.pkl', 'wb') as f:
                pickle.dump(id_counter, f)

            st.write("Model trained successfully!")
        else:
            st.write("No face detected in the uploaded image.")
