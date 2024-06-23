import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt
from tqdm import tqdm

# Direktori dataset
dataset_dir = 'Gambar'

# Parameter model
img_height, img_width = 150, 150
batch_size = 64
epochs = 20
learning_rate = 0.001

# Fungsi untuk memuat gambar dengan progres bar
def load_images_from_folder(folder):
    print("Proses baca data Gambar")
    images = []
    labels = []
    for label in tqdm(os.listdir(folder), desc="Labels", unit="label"):
        label_path = os.path.join(folder, label)
        if os.path.isdir(label_path):
            for img_filename in tqdm(os.listdir(label_path), desc=f"{label}", unit="file"):
                img_path = os.path.join(label_path, img_filename)
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.resize(img, (img_width, img_height))
                    images.append(img)
                    labels.append(label)
    return images, labels

# Memuat dataset
images, labels = load_images_from_folder(dataset_dir)
images = np.array(images)
labels = np.array(labels)

# Mengubah label menjadi one-hot encoding
lb = LabelBinarizer()
labels = lb.fit_transform(labels)

# Membagi dataset menjadi data pelatihan dan pengujian
x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Normalisasi data
x_train = x_train / 255.0
x_test = x_test / 255.0

# Membangun model CNN
model = Sequential([
    Conv2D(16, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(len(lb.classes_), activation='softmax')
])

# Kompilasi model
model.compile(optimizer=Adam(learning_rate=learning_rate),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Menampilkan ringkasan model
model.summary()

# Melatih model
history = model.fit(
    x_train, y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(x_test, y_test)
)

# Evaluasi model
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Validation Loss: {loss}")
print(f"Validation Accuracy: {round(accuracy*100,3)}")

# Menyimpan model
model.save('mangrove_CNN.h5')

# Plotting training history
# Plot training & validation accuracy values
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.show()
