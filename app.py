import os
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, Response
import mediapipe as mp
import threading
import queue
import time


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppresses INFO and WARNING, shows only ERROR


app = Flask(__name__)

# Configuration
MODEL_PATH = "hand_sign_model.h5"
CLASS_NAMES_PATH = "class_names.txt"
MIN_CONFIDENCE = 0.4
NO_HAND_TIMEOUT = 1.5

# Load model and class names
model = tf.keras.models.load_model(MODEL_PATH)
with open(CLASS_NAMES_PATH) as f:
    LABELS = [line.strip() for line in f.readlines()]

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Speech system
speech_queue = queue.Queue()
speech_thread = None
current_spoken_label = None
last_hand_detected_time = time.time()

def speech_worker():
    import pyttsx3
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    while True:
        label = speech_queue.get()
        if label is None:
            break
        engine.say(label)
        engine.runAndWait()
        speech_queue.task_done()

def start_speech_thread():
    global speech_thread
    if speech_thread is None or not speech_thread.is_alive():
        speech_thread = threading.Thread(target=speech_worker, daemon=True)
        speech_thread.start()

def speak(label):
    global current_spoken_label
    if label != current_spoken_label:
        current_spoken_label = label
        speech_queue.put(label)

def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (64, 64))
    normalized = resized.astype('float32') / 255.0
    return np.expand_dims(normalized, axis=(0, -1))

def generate_frames():
    global last_hand_detected_time, current_spoken_label
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    while True:
        success, frame = cap.read()
        if not success:
            print("Failed to capture frame")
            break
            
        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Hand detection
        results = hands.process(rgb_frame)
        hand_detected = False
        
        if results.multi_hand_landmarks:
            last_hand_detected_time = time.time()
            hand_detected = True
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        # Reset spoken label if no hand detected for timeout period
        if not hand_detected and (time.time() - last_hand_detected_time) > NO_HAND_TIMEOUT:
            current_spoken_label = None
        
        # Prediction
        if hand_detected:
            processed_frame = preprocess_frame(frame)
            predictions = model.predict(processed_frame, verbose=0)[0]
            confidence = np.max(predictions)
            predicted_label = LABELS[np.argmax(predictions)]
            
            if confidence > MIN_CONFIDENCE:
                # Display prediction
                cv2.putText(frame, f"{predicted_label} ({confidence:.2f})", 
                            (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                speak(predicted_label)
        else:
            cv2.putText(frame, "No hand detected", (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Encode frame
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
            
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    
    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), 
                   mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    start_speech_thread()
    try:
        app.run(host='0.0.0.0', port=5000, debug=True)  # This makes the app accessible on localhost
    finally:
        speech_queue.put(None)
        if speech_thread:
            speech_thread.join()
