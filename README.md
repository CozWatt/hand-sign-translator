# hand-sign-translator

The **Hand Sign Translator** is a real-time web-based application that detects hand signs using a webcam and translates them into text and speech. Built with Flask, MediaPipe, TensorFlow, and OpenCV, this app can be used to assist individuals with hearing or speech impairments.

---

# Features

- Real-time hand detection using MediaPipe
- Hand sign classification with a trained CNN model
- Speech output using `pyttsx3`
- Live webcam feed with OpenCV
- Web-based interface using Flask

---

# 📂 Project Structure
 HandSignTranslator/
│
├── app.py # Main Flask app
├── hand_sign_model.h5 # Trained CNN model
├── class_names.txt # List of class labels
├── templates/
│ └── index.html # Frontend HTML page
├── preprocess_images.py # (Preprocessing script - optional)
├── split_dataset.py # (Dataset splitting script - optional)
├── train_cnn_model.py # (Model training script)
└── README.md # Project documentation


---

## ⚙️ Requirements

# Install dependencies using pip:
  pip install -r requirements.txt

# If you don’t have a requirements.txt, here are the main packages:
  pip install flask opencv-python mediapipe tensorflow pyttsx3

# Model
  The CNN model (hand_sign_model.h5) is trained to recognize hand signs.

  Class labels are stored in class_names.txt, one per line.


# How to Run
  Clone the repository or copy the files.

  Start the app:
  1) python app.py
  2) Open in browser:
  3) Navigate to http://localhost:5000 in your web browser.


# How It Works
  1) Captures webcam frames using OpenCV
  2) Detects hands with MediaPipe
  3) Preprocesses the hand region
  4) Predicts the sign using the loaded TensorFlow model
  5) Displays the prediction on-screen
  6) Converts the prediction to speech using pyttsx3

# Note : 
  1) Make sure your webcam is accessible.
  2) You can improve accuracy by training a better model.
  3) Extend it with gesture control or multilingual support!

# License
  -> This project is for educational purposes and open to improvement and extension.

# Author
  -> Dawood Anas
  -> B.Tech CSE(AI), Annamacharya Institute Of Technology and Science

---------------------------------------------------------------------- THANK YOU :) ----------------------------------------------------------------------------



