from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import sqlite3
import os
import base64
import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from datetime import datetime

app = Flask(__name__)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
if not os.path.exists(face_cascade_path):
    logger.error(f"Haar Cascade file not found at {face_cascade_path}")
    raise FileNotFoundError(f"Haar Cascade file not found at {face_cascade_path}")
face_cascade = cv2.CascadeClassifier(face_cascade_path)
if face_cascade.empty():
    logger.error("Failed to load Haar Cascade classifier")
    raise ValueError("Failed to load Haar Cascade classifier")
recognizer = cv2.face.LBPHFaceRecognizer_create()
registration_sessions = {}

EMAIL_ADDRESS = "maraganimadhu2@gmail.com"  # Replace with your Gmail address
EMAIL_PASSWORD = "zkdx kihp zwep foto"  # Replace with Gmail App Password
ADMIN_EMAIL = "madhumaragani1@gmail.com"  # Replace with admin email

def send_email_alert(emp_id, image_path=None):
    try:
        msg = MIMEMultipart()
        msg['From'] = EMAIL_ADDRESS
        msg['To'] = ADMIN_EMAIL
        msg['Subject'] = "Unauthorized Access Attempt"
        
        body = f"Unknown person attempted to login with ID: {emp_id}\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        msg.attach(MIMEText(body, 'plain'))

        if image_path and os.path.exists(image_path):
            with open(image_path, 'rb') as f:
                img_data = f.read()
            image = MIMEImage(img_data, name=os.path.basename(image_path))
            msg.attach(image)
            logger.debug(f"Attached image {image_path} to email")

        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        server.send_message(msg)
        server.quit()
        logger.info("Email alert sent successfully")
    except Exception as e:
        logger.error(f"Failed to send email alert: {e}")

def init_db():
    db_path = os.path.abspath('faces.db')
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS employees 
                 (id TEXT PRIMARY KEY, name TEXT, model BLOB)''')
    conn.commit()
    conn.close()
    logger.info("Checked/Initialized faces.db")

def detect_and_crop_face(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=2)
    if len(faces) == 0:
        logger.warning("No faces detected in image")
        return None
    if len(faces) > 1:
        logger.warning(f"Multiple faces detected ({len(faces)}), using first one")
    (x, y, w, h) = faces[0]
    face_gray = gray[y:y+h, x:x+w]
    face_resized = cv2.resize(face_gray, (100, 100))
    logger.debug(f"Face detected at ({x}, {y}, {w}, {h}), resized to 100x100")
    return face_resized

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET'])
def register_page():
    return render_template('register.html')

@app.route('/register_batch', methods=['POST'])
def register_batch():
    emp_id = request.form.get('id', '').strip().lower()
    name = request.form.get('name')
    batch_num = int(request.form.get('batch_num', 0))
    images_data = request.form.getlist('images[]')
    
    logger.debug(f"Received: id={emp_id}, name={name}, batch_num={batch_num}, images_count={len(images_data)}")
    
    if not emp_id or not name or not images_data:
        logger.error(f"Missing data: id={emp_id}, name={name}, images_count={len(images_data)}")
        return jsonify({'error': 'Missing ID, name, or images'}), 400
    
    if emp_id not in registration_sessions:
        registration_sessions[emp_id] = {'name': name, 'faces': []}
    
    session = registration_sessions[emp_id]
    if session['name'] != name:
        return jsonify({'error': 'Name mismatch for ID'}), 400
    
    for i, image_data in enumerate(images_data):
        try:
            img_data = base64.b64decode(image_data.split(',')[1])
            img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                logger.error(f"Failed to decode image {i} in batch {batch_num}")
                return jsonify({'error': f'Invalid image data in batch {batch_num}, sample {i}'}), 400
            logger.debug(f"Image {i} shape: {img.shape}, dtype: {img.dtype}")

            debug_path = os.path.join(app.config['UPLOAD_FOLDER'], f"debug_batch_{batch_num}_img_{i}.jpg")
            cv2.imwrite(debug_path, img)
            logger.debug(f"Saved raw image {i} to {debug_path}")

            face = detect_and_crop_face(img)
            if face is None:
                logger.warning(f"No face detected in image {i}, saved to {debug_path}")
                return jsonify({'error': f'No face detected in batch {batch_num}, sample {i}. Ensure your face is centered and well-lit.'}), 400
            session['faces'].append(face)
        except Exception as e:
            logger.error(f"Error processing image {i}: {e}")
            return jsonify({'error': 'Invalid image data in batch {batch_num}, sample {i}'}), 400
    
    total_samples = len(session['faces'])
    if total_samples >= 200:
        conn = sqlite3.connect('faces.db')
        c = conn.cursor()
        c.execute("SELECT id FROM employees WHERE id=?", (emp_id,))
        if c.fetchone():
            conn.close()
            del registration_sessions[emp_id]
            return jsonify({'error': 'Employee ID already exists'}), 400
        
        labels = np.array([0] * total_samples)
        try:
            recognizer.train(session['faces'], labels)
        except Exception as e:
            logger.error(f"Failed to train recognizer: {e}")
            return jsonify({'error': 'Failed to train face model'}), 500
        
        model_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{emp_id}_model.yml")
        try:
            recognizer.write(model_path)
            with open(model_path, 'rb') as f:
                model_data = f.read()
            test_recognizer = cv2.face.LBPHFaceRecognizer_create()
            test_recognizer.read(model_path)
            logger.debug(f"Model for ID {emp_id} validated successfully")
        except Exception as e:
            logger.error(f"Failed to save or validate model for ID {emp_id}: {e}")
            if os.path.exists(model_path):
                os.remove(model_path)
            return jsonify({'error': 'Failed to save face model'}), 500
        
        c.execute("INSERT INTO employees (id, name, model) VALUES (?, ?, ?)", 
                  (emp_id, name, model_data))
        conn.commit()
        conn.close()
        if os.path.exists(model_path):
            os.remove(model_path)
        del registration_sessions[emp_id]
        logger.info(f"Registered employee: {name} with ID: {emp_id} - Registration fully completed")
        return jsonify({'message': 'Registration successful', 'complete': True})
    
    logger.info(f"Processed batch {batch_num} for ID: {emp_id}, total samples: {total_samples}")
    return jsonify({'message': f'Batch {batch_num} processed', 'complete': False, 'total_samples': total_samples})

@app.route('/verify', methods=['GET', 'POST'])
def verify_face():
    if request.method == 'GET':
        return render_template('verify.html')
    
    emp_id = request.form.get('id', '').strip().lower()
    image_data = request.form.get('image')
    
    if not emp_id or not image_data:
        return jsonify({'error': 'Missing ID or image'}), 400
    
    try:
        img_data = base64.b64decode(image_data.split(',')[1])
        img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            logger.error("Failed to decode verification image")
            return jsonify({'error': 'Invalid image data'}), 400
        
        debug_path = os.path.join(app.config['UPLOAD_FOLDER'], f"verify_{emp_id}_{int(datetime.now().timestamp())}.jpg")
        cv2.imwrite(debug_path, img)
        logger.debug(f"Saved verification image to {debug_path}")
        
        face = detect_and_crop_face(img)
        if face is None:
            logger.warning(f"No face detected for verification, ID: {emp_id}")
            return jsonify({'error': 'No face detected. Ensure your face is centered, well-lit, and matches registration conditions.'}), 400
    except Exception as e:
        logger.error(f"Error processing verification image: {e}")
        return jsonify({'error': 'Invalid image data'}), 400
    
    conn = sqlite3.connect('faces.db')
    c = conn.cursor()
    c.execute("SELECT id FROM employees")
    registered_ids = [row[0] for row in c.fetchall()]
    logger.debug(f"Registered IDs in database: {registered_ids}")
    
    c.execute("SELECT name, model FROM employees WHERE id=?", (emp_id,))
    result = c.fetchone()
    if not result:
        conn.close()
        logger.warning(f"ID {emp_id} not found in database")
        send_email_alert(emp_id, debug_path)
        return jsonify({'error': 'Employee ID not found'}), 404
    
    name, model_data = result
    model_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{emp_id}_temp_model.yml")
    try:
        with open(model_path, 'wb') as f:
            f.write(model_data)
        recognizer.read(model_path)
        logger.debug(f"Successfully loaded model for ID {emp_id}")
    except Exception as e:
        logger.error(f"Failed to load model for ID {emp_id}: {e}")
        if os.path.exists(model_path):
            os.remove(model_path)
        conn.close()
        return jsonify({'error': 'Failed to load face model'}), 500
    finally:
        if os.path.exists(model_path):
            os.remove(model_path)
    
    confidence_threshold = 60
    try:
        label, confidence = recognizer.predict(face)
        logger.info(f"Prediction for ID {emp_id}: label={label}, confidence={confidence:.2f}")
        
        if confidence > confidence_threshold:
            logger.info(f"Face mismatch for ID: {emp_id}, confidence: {confidence:.2f}")
            conn.close()
            send_email_alert(emp_id, debug_path)
            return jsonify({'error': 'Face does not match registered employee. Ensure similar lighting and angle as during registration.'}), 401
    except Exception as e:
        logger.error(f"Prediction failed for ID {emp_id}: {e}")
        conn.close()
        return jsonify({'error': 'Face recognition failed'}), 500
    
    conn.close()
    logger.info(f"Successful verification for {name} (ID: {emp_id}), confidence: {confidence:.2f}")
    return jsonify({
        'message': f'Access granted for {name}'
    })

if __name__ == '__main__':
    init_db()
    app.run(debug=False, threaded=True, host='0.0.0.0', port=5000)