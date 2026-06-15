# train_on_railway.py
import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import gdown
import zipfile

print("="*50)
print("TRAINING MODEL DI RAILWAY")
print("="*50)

# Download dataset dari Google Drive (upload dataset Anda ke Google Drive dulu)
# Atau gunakan dataset kecil untuk testing
print("Mendownload dataset...")

# Contoh: download dataset kecil dari URL
# gdown.download("YOUR_DATASET_URL", "dataset.zip")

# Ekstrak dataset
# with zipfile.ZipFile("dataset.zip", 'r') as zip_ref:
#     zip_ref.extractall("dataset")

# Atau gunakan dataset yang sudah ada di repository
TRAIN_DIR = 'raw-img'  # Sesuaikan dengan struktur folder Anda

if not os.path.exists(TRAIN_DIR):
    print("ERROR: Dataset tidak ditemukan!")
    print("Silakan upload dataset ke Railway atau gunakan Google Drive")
    exit(1)

# Parameter
IMG_HEIGHT, IMG_WIDTH = 150, 150
BATCH_SIZE = 32
EPOCHS = 10  # Kurangi epoch agar cepat

# Data Generator
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    horizontal_flip=True,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

CLASS_NAMES = list(train_generator.class_indices.keys())
print(f"Kelas: {CLASS_NAMES}")

# Membangun model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(len(CLASS_NAMES), activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print("Mulai training...")
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator
)

# Simpan model
os.makedirs('models', exist_ok=True)
model.save('models/model_hewan_cnn.h5')
print("✅ Model saved to models/model_hewan_cnn.h5")

# Simpan class names
import json
with open('models/class_names.json', 'w') as f:
    json.dump(CLASS_NAMES, f)

print("✅ Training selesai!")