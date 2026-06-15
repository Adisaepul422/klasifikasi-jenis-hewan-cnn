"""
train_model.py - Klasifikasi Hewan dengan CNN (Tanpa Venv)
"""

import os
import shutil
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Import TensorFlow
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ===================== KONFIGURASI =====================
IMG_HEIGHT = 150
IMG_WIDTH = 150
BATCH_SIZE = 32
EPOCHS = 20

# Mapping nama folder Italia -> Inggris
ITALIAN_TO_ENGLISH = {
    'cane': 'Anjing',
    'gatto': 'Kucing',
    'cavallo': 'Kuda',
    'elefante': 'Gajah',
    'farfalla': 'Kupu-kupu',
    'gallina': 'Ayam',
    'mucca': 'Sapi',
    'pecora': 'Domba',
    'ragno': 'Laba-laba',
    'scoiattolo': 'Tupai'
}

# Path dataset (folder raw-img)
RAW_DATA_DIR = 'raw-img'

# ===================== SPLIT DATA =====================
print("="*50)
print("MEMBAGI DATA MENJADI TRAIN DAN VALIDATION...")
print("="*50)

TRAIN_DIR = 'dataset/train'
VALIDATION_DIR = 'dataset/validation'

# Hapus folder lama jika ada
if os.path.exists(TRAIN_DIR):
    shutil.rmtree(TRAIN_DIR)
if os.path.exists(VALIDATION_DIR):
    shutil.rmtree(VALIDATION_DIR)

os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(VALIDATION_DIR, exist_ok=True)

# Loop setiap kelas
for italian_name, english_name in ITALIAN_TO_ENGLISH.items():
    class_path = os.path.join(RAW_DATA_DIR, italian_name)
    
    if not os.path.exists(class_path):
        print(f"⚠️ Folder tidak ditemukan: {class_path}")
        continue
    
    images = [f for f in os.listdir(class_path) 
              if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    if len(images) == 0:
        print(f"⚠️ Tidak ada gambar di folder: {class_path}")
        continue
    
    # Split 80% train, 20% validation
    train_images, val_images = train_test_split(images, test_size=0.2, random_state=42)
    
    train_class_dir = os.path.join(TRAIN_DIR, english_name)
    val_class_dir = os.path.join(VALIDATION_DIR, english_name)
    
    os.makedirs(train_class_dir, exist_ok=True)
    os.makedirs(val_class_dir, exist_ok=True)
    
    for img in train_images:
        shutil.copy2(os.path.join(class_path, img), os.path.join(train_class_dir, img))
    
    for img in val_images:
        shutil.copy2(os.path.join(class_path, img), os.path.join(val_class_dir, img))
    
    print(f"✅ {italian_name} ({english_name}): Train={len(train_images)}, Val={len(val_images)}")

print("\n✅ Data berhasil dibagi!")

# ===================== DATA AUGMENTATION =====================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(rescale=1./255)

# ===================== LOAD DATA =====================
print("\n" + "="*50)
print("LOADING DATASET...")
print("="*50)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_directory(
    VALIDATION_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

CLASS_NAMES = list(train_generator.class_indices.keys())
print(f"\n📋 Kelas yang terdeteksi: {CLASS_NAMES}")
print(f"📊 Total gambar training: {train_generator.samples}")
print(f"📊 Total gambar validasi: {validation_generator.samples}")

# ===================== MEMBANGUN MODEL CNN =====================
print("\n" + "="*50)
print("MEMBANGUN MODEL CNN...")
print("="*50)

model = models.Sequential([
    layers.Conv2D(16, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(len(CLASS_NAMES), activation='softmax')
])

# Simpan model dengan kompresi
model.save('models/model_hewan_cnn.h5', save_format='h5', include_optimizer=False)

# ===================== KOMPILASI MODEL =====================
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ===================== TRAINING =====================
print("\n" + "="*50)
print("MULAI TRAINING...")
print("="*50)

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE,
    verbose=1
)

# ===================== SIMPAN MODEL =====================
os.makedirs('models', exist_ok=True)
model.save('models/model_hewan_cnn.h5')
print("\n✅ Model berhasil disimpan ke 'models/model_hewan_cnn.h5'")

# Simpan class names
import json
with open('models/class_names.json', 'w') as f:
    json.dump(CLASS_NAMES, f)
print("✅ Class names disimpan ke 'models/class_names.json'")

# ===================== PLOT HASIL =====================
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('models/training_history.png')
plt.show()

# ===================== EVALUASI =====================
test_loss, test_acc = model.evaluate(validation_generator, verbose=0)
print(f"\n🎯 Akurasi pada data validasi: {test_acc:.2%}")
print(f"📉 Loss pada data validasi: {test_loss:.4f}")