






import os
import cv2
import numpy as np
import pickle
from flask import Flask, request, jsonify
from sklearn.svm import SVC
from datetime import datetime
from collections import Counter
import insightface
from insightface.app import FaceAnalysis
import psycopg2
from psycopg2.pool import SimpleConnectionPool
from psycopg2.extras import execute_values, execute_batch
import json
import logging
from functools import lru_cache
from sklearn.neighbors import NearestNeighbors

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)

model = cv2.FaceDetectorYN.create(
    "face_detection_yunet_2023mar.onnx",
    "",
    (320, 320),
    0.6,
    0.3,
    5000
)

# Initialize ArcFace model
app_face = FaceAnalysis()
app_face.prepare(ctx_id=0, det_size=(640, 640))

# PostgreSQL connection details
DB_NAME = "Attendx"
DB_USER = "postgres"
DB_PASSWORD ="1234"
DB_HOST = "localhost"
DB_PORT = "5433"

# Create a connection pool
pool = SimpleConnectionPool(1, 20,
    dbname=DB_NAME,
    user=DB_USER,
    password=DB_PASSWORD,
    host=DB_HOST,
    port=DB_PORT
)

def get_db_connection():
    return pool.getconn()

def return_db_connection(conn):
    pool.putconn(conn)

# Create tables if they don't exist
def create_tables():
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id SERIAL PRIMARY KEY,
                    user_id VARCHAR(255) UNIQUE NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            cur.execute('''
                CREATE TABLE IF NOT EXISTS encodings (
                    id SERIAL PRIMARY KEY,
                    user_id VARCHAR(255) REFERENCES users(user_id),
                    encoding BYTEA NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            cur.execute('''
                CREATE TABLE IF NOT EXISTS models (
                    id SERIAL PRIMARY KEY,
                    name VARCHAR(255) UNIQUE NOT NULL,
                    model_data BYTEA NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Add indexing
            cur.execute('CREATE INDEX IF NOT EXISTS idx_encodings_user_id ON encodings(user_id)')
            
        conn.commit()
        logging.info("Tables created successfully")
    except psycopg2.Error as e:
        logging.error(f"Error creating tables: {e}")
        conn.rollback()
    finally:
        return_db_connection(conn)

# Call this function to create tables when the app starts
create_tables()

@lru_cache(maxsize=1000)
def load_image_from_path(file_path):
    try:
        img = cv2.imread(file_path)
        if img is None:
            print(f"Failed to load image: {file_path}")
            return None
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(f"Error loading image {file_path}: {str(e)}")
        return None

def batch_find_encodings(images, class_name):
    encodeList = []
    for img in images:
        faces = app_face.get(img)
        for face in faces:
            if face:
                encoding = face.embedding
                encodeList.append({'encoding': encoding, 'name': class_name})
            else:
                print(f"No face found in an image for {class_name}. Skipping...")
    return encodeList

def update_encodings(new_encodings, user_folder):
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            # Insert user if not exists
            cur.execute("INSERT INTO users (user_id) VALUES (%s) ON CONFLICT (user_id) DO NOTHING", (user_folder,))
            
            # Remove existing encodings for this user
            cur.execute("DELETE FROM encodings WHERE user_id = %s", (user_folder,))
            
            # Batch insert new encodings
            execute_batch(cur, 
                          "INSERT INTO encodings (user_id, encoding) VALUES (%s, %s)",
                          [(user_folder, pickle.dumps(enc['encoding'])) for enc in new_encodings])
        
        conn.commit()
        print(f'Encodings updated for user {user_folder}')
    except Exception as e:
        conn.rollback()
        print(f"Error updating encodings: {e}")
    finally:
        return_db_connection(conn)
    return new_encodings

class SingleClassClassifier:
    def __init__(self, single_class):
        self.single_class = single_class
    
    def predict(self, X):
        return np.array([self.single_class] * len(X))
    
    def predict_proba(self, X):
        return np.array([[0, 1]] * len(X))  # Always predict 100% probability for the single class

def train_svm_classifier():
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT user_id, encoding FROM encodings")
            rows = cur.fetchall()
        
        if rows:
            X = []
            y = []
            for user_id, encoding_bytes in rows:
                X.append(pickle.loads(encoding_bytes))
                y.append(user_id)
            
            X = np.array(X)
            y = np.array(y)
            
            unique_classes = np.unique(y)
            if len(unique_classes) > 1:
                svm_clf = SVC(probability=True, kernel='linear')
                svm_clf.fit(X, y)
            else:
                svm_clf = SingleClassClassifier(unique_classes[0])
            
            # Save the model to the database
            model_bytes = pickle.dumps(svm_clf)
            with conn.cursor() as cur:
                cur.execute("INSERT INTO models (name, model_data) VALUES (%s, %s) ON CONFLICT (name) DO UPDATE SET model_data = EXCLUDED.model_data",
                            ('svm_model', model_bytes))
            conn.commit()
            print("SVM classifier updated and saved to database")
        else:
            print("No encodings found in the database.")
            svm_clf = None
    except Exception as e:
        conn.rollback()
        print(f"Error training SVM classifier: {e}")
        svm_clf = None
    finally:
        return_db_connection(conn)
    
    return svm_clf

@lru_cache(maxsize=1000)
def is_user_registered(user_id):
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM users WHERE user_id = %s", (user_id,))
            user_exists = cur.fetchone() is not None
    finally:
        return_db_connection(conn)
    
    return user_exists

def register_user(user_id):
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("INSERT INTO users (user_id) VALUES (%s) ON CONFLICT (user_id) DO NOTHING", (user_id,))
        conn.commit()
    finally:
        return_db_connection(conn)

def find_matching_users(face_encodings):
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT user_id, encoding FROM encodings")
            rows = cur.fetchall()
    finally:
        return_db_connection(conn)
    
    # Use NearestNeighbors for efficient similarity search
    X = np.array([pickle.loads(row[1]) for row in rows])
    user_ids = [row[0] for row in rows]
    nn = NearestNeighbors(n_neighbors=1, metric='cosine')
    nn.fit(X)
    
    matching_users = []
    for face_encoding in face_encodings:
        distances, indices = nn.kneighbors([face_encoding])
        if distances[0][0] < 0.5:  # Adjust this threshold as needed
            matching_users.append(user_ids[indices[0][0]])
    
    return matching_users

# Your existing route handlers (detect-face, upload, detect, mark_attendance, pre_upload) remain the same
@app.route('/upload', methods=['POST'])
def upload_image():
    data = request.json
    if not data or 'urls' not in data or 'user' not in data:
        return jsonify({'error': 'Missing file paths or user data'}), 400

    file_paths = data['urls']
    user_folder = data['user']

    # Check if user is already registered
    if is_user_registered(user_folder):
        return jsonify({'message': f'User {user_folder} already exists. No update performed.'}), 200

    images = []
    for file_path in file_paths:
        img = load_image_from_path(file_path)
        if img is not None:
            images.append(img)
        else:
            logging.warning(f"Failed to load image: {file_path}")

    if not images:
        return jsonify({'error': 'Failed to load any images from provided paths'}), 400

    try:
        new_encodings = batch_find_encodings(images, user_folder)
        if not new_encodings:
            return jsonify({'error': 'No faces found in the provided images'}), 400
        
        update_encodings(new_encodings, user_folder)
        
        # Only train the classifier if it's a new user
        classifier = train_svm_classifier()
        
        if classifier is None:
            return jsonify({'error': 'Failed to train classifier. Check your image data.'}), 500

        # Register the user
        register_user(user_folder)

        return jsonify({'message': 'Encodings updated and model saved successfully'}), 200
    except Exception as e:
        logging.error(f"Error in upload_image: {str(e)}")
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500
    
    
@app.route('/detect', methods=['POST'])
def detect_face():
    data = request.json
    if not data or 'url' not in data:
        return jsonify({'error': 'Missing image file path'}), 400

    img_path = data['url']
    img = load_image_from_path(img_path)
    if img is None:
        return jsonify({'error': 'Failed to load image from file path'}), 400

    conn = get_db_connection()
    cur = conn.cursor()
    
    cur.execute("SELECT model_data FROM models WHERE name = 'svm_model'")
    model_data = cur.fetchone()
    
    if model_data:
        svm_clf = pickle.loads(model_data[0])
    else:
        return jsonify({'error': 'SVM model not found in the database'}), 500

    cur.close()
    conn.close()

    message = process_image(img, svm_clf)
    return jsonify({'message': message}), 200

def process_image(img, svm_clf):
    faces = app_face.get(img)
    if not faces:
        return "No face detected in the image"

    face_encoding = faces[0].embedding
    predictions = svm_clf.predict_proba([face_encoding])
    best_match_index = np.argmax(predictions[0])
    best_match_probability = predictions[0][best_match_index]

    if best_match_probability > 0.5:  # You can adjust this threshold
        name = svm_clf.classes_[best_match_index]
        return f"Face recognized: {name}"
    else:
        return "Unknown person"

@app.route('/mark_attendance', methods=['POST'])
def mark_attendance():
    data = request.json
    if not data or 'url' not in data or 'user_id' not in data:
        return jsonify({'error': 'Missing image file path or user ID'}), 400

    img_path = data['url']
    user_id = data['user_id']

    img = load_image_from_path(img_path)
    if img is None:
        return jsonify({'error': 'Failed to load image from file path'}), 400

    faces = app_face.get(img)
    if not faces:
        return jsonify({'error': 'No face detected in the uploaded image'}), 400

    uploaded_encoding = faces[0].embedding

    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute("SELECT encoding FROM encodings WHERE user_id = %s", (user_id,))
    rows = cur.fetchall()

    if not rows:
        cur.close()
        conn.close()
        return jsonify({'message': f'User ID {user_id} does not exist'}), 404

    match_found = False
    for row in rows:
        known_encoding = pickle.loads(row[0])
        similarity = np.dot(known_encoding, uploaded_encoding) / (np.linalg.norm(known_encoding) * np.linalg.norm(uploaded_encoding))
        if similarity > 0.5:  # Adjust this threshold as needed
            match_found = True
            break

    cur.close()
    conn.close()

    if match_found:
        return jsonify({'recognized': True, 'message': f'Face recognized for user {user_id}'}), 200
    else:
        return jsonify({'recognized': False, 'message': 'Face does not match the user ID'}), 404
    
    
    
@app.route('/pre_upload', methods=['POST'])
def pre_upload():
    data = request.json
    if not data or 'url' not in data or 'user' not in data:
        return jsonify({'success': False, 'message': 'Invalid request'}), 400

    file_path = data['url']
    user_id = data['user']

    img = load_image_from_path(file_path)
    if img is None:
        return jsonify({'success': False, 'message': 'Invalid image'}), 400

    faces = app_face.get(img)
    if not faces:
        return jsonify({'success': False, 'message': 'No face detected'}), 400

    face_encodings = [face.embedding for face in faces]
    user_exists = is_user_registered(user_id)
    matching_users = find_matching_users(face_encodings)

    if not user_exists and not matching_users:
        return jsonify({'success': False, 'message': 'User not found'}), 404
    elif user_exists and not matching_users:
        return jsonify({'success': False, 'message': 'Face mismatch'}), 422
    elif not user_exists and matching_users:
        return jsonify({'success': False, 'message': 'Account mismatch'}), 422
    else:
        if user_id in matching_users:
            return jsonify({'success': True, 'message': 'Verification successful'}), 200
        else:
            return jsonify({'success': False, 'message': 'Account mismatch'}), 409

if __name__ == '__main__':
    app.run(debug=False)
   









