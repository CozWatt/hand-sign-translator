# ✋ Hand Sign Translator

The **Hand Sign Translator** is a real-time, web-based application that uses a webcam to detect hand signs and translate them into both **text and speech**. It is built with **Flask**, **MediaPipe**, **TensorFlow**, and **OpenCV**, and is designed to assist individuals with hearing or speech impairments.

---

## 🚀 Features

- 📸 Real-time hand detection using **MediaPipe**
- 🤖 Hand sign classification with a **trained CNN model**
- 🔊 Speech output using **pyttsx3**
- 🖥️ Live webcam feed with **OpenCV**
- 🌐 User-friendly web interface built with **Flask**

---

## 📁 Project Structure

```
HandSignTranslator/
│
├── app.py                 # Main Flask application
├── hand_sign_model.h5     # Trained CNN model
├── class_names.txt        # List of hand sign labels (classes)
├── templates/
│   └── index.html         # Frontend HTML page
├── preprocess_images.py   # Optional image preprocessing script
├── split_dataset.py       # Optional dataset splitting script
├── train_cnn_model.py     # Script for training the CNN model
└── README.md              # Project documentation
```

---

## ⚙️ Installation

### 🔧 1. Install Dependencies

If you have a `requirements.txt`:

```bash
pip install -r requirements.txt
```

Or manually install the main packages:

```bash
pip install flask opencv-python mediapipe tensorflow pyttsx3
```

---

## 🧠 Model Info

- The CNN model (`hand_sign_model.h5`) is trained to recognize hand signs.
- The corresponding class labels are stored in `class_names.txt` (one per line).

---

## 🏃 How to Run the App

1. **Clone the repository** or copy the files to your system.
2. **Run the Flask server:**

```bash
python app.py
```

3. **Open the app in your browser:**

```
http://localhost:5000
```

---

## 🔍 How It Works

1. Captures video frames from the webcam using **OpenCV**
2. Detects hands in the frame using **MediaPipe**
3. Extracts and preprocesses the hand region
4. Feeds the preprocessed image into the **trained TensorFlow model**
5. Displays the predicted sign on the screen
6. Converts the prediction into speech using **pyttsx3**

---

## 📝 Notes

- 🎥 Ensure your **webcam** is connected and accessible.
- 📈 You can improve prediction accuracy by retraining the model with better or more data.
- 💡 Future improvements:
  - Add **gesture control**
  - Add **multilingual support**
  - Add **user authentication or prediction history**

---

## 📄 License

This project is intended for **educational purposes** and is open to community contributions and enhancements.

---

## 👤 Author

**Dawood Anas**  


---

###  Thank You for Visiting!
