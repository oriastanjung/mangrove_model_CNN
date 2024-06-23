import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt

# Parameter model
img_height, img_width = 150, 150

# Direktori model yang sudah disimpan
model_path = 'mangrove_CNN.h5'

# Load model yang sudah dilatih
model = load_model(model_path)

# Label binarizer (harus sama dengan yang digunakan saat pelatihan)
labels = ['Avicennia alba', 'Bruguiera cylindrica', 'Bruguiera gymnorrhiza',
          'Lumnitzora littorea', 'Rhizophora apiculata', 'Rhizophora mucronata',
          'Scyphiphora hydrophyllacea', 'Senoratia alba', 'Xylocarpus granatum']

lb = LabelBinarizer()
lb.fit(labels)

# # Fungsi untuk memuat dan memproses gambar baru
# def load_and_process_image(img_path):
#     img = cv2.imread(img_path)
#     img = cv2.resize(img, (img_width, img_height))
#     img = img / 255.0
#     img = np.expand_dims(img, axis=0)  # Menambahkan dimensi batch
#     return img

# # Fungsi untuk melakukan prediksi
# def predict_image_class(img_path):
#     img = load_and_process_image(img_path)
#     predictions = model.predict(img)
#     predicted_class_index = np.argmax(predictions, axis=1)
#     predicted_class = lb.classes_[predicted_class_index][0]
#     confidence = np.max(predictions)
#     return predicted_class, confidence

# # Path gambar baru untuk diprediksi
# # new_image_path = 'Gambar_Uji/Avicenia alba.jpeg'
# new_image_path = 'Gambar_Uji/avicennia_alba76.jpg'
# # new_image_path = 'Gambar_Uji/xylocarpus_granatum72.jpg'

# # Melakukan prediksi pada gambar baru
# predicted_class, confidence = predict_image_class(new_image_path)
# print(f"Predicted class: {predicted_class}")
# print(f"Confidence: {confidence:.2f}")

# # Fungsi untuk menampilkan gambar dan prediksi
# def display_image_with_prediction(img_path, predicted_class, confidence):
#     img = cv2.imread(img_path)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     plt.imshow(img)
#     plt.title(f"Predicted: {predicted_class} ({confidence:.2f})")
#     plt.axis('off')
#     plt.show()

# # Menampilkan gambar dengan prediksi
# display_image_with_prediction(new_image_path, predicted_class, confidence)



def klasifikasiMangrove(image):
    img = image
    # img = cv2.imread(img_path)
    img = cv2.resize(img, (img_width, img_height))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    predictions = model.predict(img)
    predicted_class_index = np.argmax(predictions, axis=1)
    predicted_class = lb.classes_[predicted_class_index][0]
    confidence = np.max(predictions)
    id_mangrove = predicted_class_index
    return predicted_class, id_mangrove, confidence
