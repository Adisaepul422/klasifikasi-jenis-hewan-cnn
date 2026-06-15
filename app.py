"""
app.py
Aplikasi web untuk klasifikasi jenis hewan menggunakan CNN
"""

import os
import sys
import numpy as np
from flask import Flask, render_template, request, jsonify, url_for
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import uuid
import logging

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Inisialisasi Flask
app = Flask(__name__)

# Konfigurasi
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Maks 16MB

# Ekstensi file yang diizinkan
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Nama kelas (10 jenis hewan)
CLASS_NAMES = ['Anjing', 'Ayam', 'Burung', 'Domba', 'Gajah', 
               'Kucing', 'Kupu', 'Laba', 'Sapi', 'Tupai']

# Info tambahan untuk setiap kelas
CLASS_INFO = {
    'Anjing': {'icon': '🐕', 'desc': 'Hewan peliharaan yang setia'},
    'Kucing': {'icon': '🐈', 'desc': 'Hewan peliharaan yang lucu'},
    'Burung': {'icon': '🐦', 'desc': 'Hewan yang bisa terbang'},
    'Gajah': {'icon': '🐘', 'desc': 'Hewan darat terbesar'},
    'Kupu': {'icon': '🦋', 'desc': 'Serangga dengan sayap indah'},
    'Ayam': {'icon': '🐔', 'desc': 'Hewan ternak penghasil telur'},
    'Sapi': {'icon': '🐄', 'desc': 'Hewan ternak penghasil susu'},
    'Domba': {'icon': '🐑', 'desc': 'Hewan ternak penghasil wol'},
    'Laba': {'icon': '🕷️', 'desc': 'Serangga pembuat jaring'},
    'Tupai': {'icon': '🐿️', 'desc': 'Hewan pemanjat yang lincah'}
}

# Global variable untuk model
model = None

def load_cnn_model():
    """Load model CNN dengan berbagai kemungkinan path"""
    global model
    
    try:
        # Log semua informasi
        print("="*50)
        print("DEBUGGING MODEL LOAD")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Files in root: {os.listdir('.')}")
        
        # Cek folder models
        if os.path.exists('models'):
            print(f"models folder exists. Contents: {os.listdir('models')}")
        else:
            print("models folder NOT FOUND!")
        
        # Coba cari file .h5 di seluruh folder
        print("\nSearching for .h5 files...")
        for root, dirs, files in os.walk('.'):
            for file in files:
                if file.endswith('.h5'):
                    print(f"Found: {os.path.join(root, file)}")
        
        print("="*50)
        
        # Path yang dicoba
        possible_paths = [
            'models/model_hewan_cnn.h5',
            './models/model_hewan_cnn.h5',
            os.path.join(os.path.dirname(__file__), 'models', 'model_hewan_cnn.h5'),
        ]
        
        for path in possible_paths:
            print(f"Checking path: {path}")
            if os.path.exists(path):
                print(f"✅ Model ditemukan di: {path}")
                model = load_model(path, compile=False)
                print("✅ Model berhasil dimuat!")
                return model
            else:
                print(f"❌ Path not found: {path}")
        
        print("❌ Model tidak ditemukan")
        return None
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

# Fungsi pengecekan file yang diizinkan
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Fungsi prediksi gambar
def predict_image(img_path):
    """Menerima path gambar, mengembalikan hasil prediksi"""
    global model
    
    if model is None:
        logger.error("Model is None, cannot predict")
        return None, None, None
    
    try:
        # Load dan preprocess gambar
        img = image.load_img(img_path, target_size=(150, 150))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        
        # Prediksi
        predictions = model.predict(img_array, verbose=0)
        predicted_idx = np.argmax(predictions[0])
        confidence = np.max(predictions[0])
        predicted_class = CLASS_NAMES[predicted_idx]
        
        logger.info(f"Prediction: {predicted_class} with confidence {confidence:.2%}")
        
        return predicted_class, confidence, predictions[0]
    
    except Exception as e:
        logger.error(f"Error in predict_image: {e}")
        return None, None, None

# ===================== ROUTES =====================

@app.route('/')
def index():
    """Halaman utama"""
    return render_template('index.html', class_names=CLASS_NAMES, class_info=CLASS_INFO)

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint untuk prediksi gambar yang diupload"""
    
    logger.info("=== PREDICT REQUEST RECEIVED ===")
    
    # Cek apakah ada file yang diupload
    if 'file' not in request.files:
        logger.warning("No file in request.files")
        return render_template('index.html', error='Tidak ada file yang dipilih', 
                              class_names=CLASS_NAMES, class_info=CLASS_INFO)
    
    file = request.files['file']
    logger.info(f"File received: {file.filename}")
    
    # Cek apakah file kosong
    if file.filename == '':
        logger.warning("Empty filename")
        return render_template('index.html', error='File kosong, pilih gambar terlebih dahulu',
                              class_names=CLASS_NAMES, class_info=CLASS_INFO)
    
    # Cek ekstensi file
    if not allowed_file(file.filename):
        logger.warning(f"Invalid file type: {file.filename}")
        return render_template('index.html', error='Format file tidak didukung. Gunakan: png, jpg, jpeg, gif',
                              class_names=CLASS_NAMES, class_info=CLASS_INFO)
    
    # Simpan file
    try:
        ext = file.filename.rsplit('.', 1)[1].lower()
        filename = f"{uuid.uuid4().hex}.{ext}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        file.save(filepath)
        logger.info(f"File saved to: {filepath}")
        
        # Lakukan prediksi
        predicted_class, confidence, all_probs = predict_image(filepath)
        
        if predicted_class is None:
            logger.error("Prediction failed - model not loaded")
            return render_template('index.html', error='Model belum dimuat. Silakan coba lagi.',
                                  class_names=CLASS_NAMES, class_info=CLASS_INFO)
        
        # Siapkan probabilitas untuk setiap kelas
        probabilities = {}
        if all_probs is not None:
            for i, class_name in enumerate(CLASS_NAMES):
                probabilities[class_name] = float(all_probs[i])
        
        logger.info(f"Prediction successful: {predicted_class}")
        
        return render_template('result.html', 
                             filename=filename,
                             filepath=filepath,
                             predicted_class=predicted_class,
                             confidence=round(confidence * 100, 2),
                             probabilities=probabilities,
                             class_info=CLASS_INFO)
    
    except Exception as e:
        logger.error(f"Error in predict route: {e}")
        return render_template('index.html', error=f'Terjadi kesalahan: {str(e)}',
                              class_names=CLASS_NAMES, class_info=CLASS_INFO)

@app.route('/about')
def about():
    """Halaman tentang aplikasi"""
    return render_template('about.html')

@app.route('/health')
def health():
    """Health check endpoint untuk Railway"""
    if model is None:
        return {'status': 'error', 'message': 'Model not loaded'}, 500
    return {'status': 'ok', 'message': 'Model loaded'}

@app.route('/clear')
def clear():
    """Membersihkan file upload"""
    import shutil
    folder = app.config['UPLOAD_FOLDER']
    if os.path.exists(folder):
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                logger.error(f"Failed to delete {file_path}: {e}")
    return "Upload folder cleared"

# ===================== MAIN =====================
if __name__ == '__main__':
    # Buat folder yang diperlukan
    os.makedirs('static/uploads', exist_ok=True)
    os.makedirs('static/css', exist_ok=True)
    
    # Load model sebelum menjalankan server
    logger.info("Loading CNN model...")
    model = load_cnn_model()
    
    if model is None:
        logger.error("WARNING: Model failed to load! Prediction will not work.")
    else:
        logger.info("Model loaded successfully!")
    
    # Jalankan aplikasi
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)