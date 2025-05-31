import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

def load_data(folder):
    images, labels = [], []
    class_names = sorted(os.listdir(folder))
    for class_idx, class_name in enumerate(class_names):
        class_path = os.path.join(folder, class_name)
        if not os.path.isdir(class_path):
            continue

        for file in os.listdir(class_path):
            if file.endswith('.npy'):
                try:
                    img = np.load(os.path.join(class_path, file))
                    images.append(img)
                    labels.append(class_idx)
                except Exception as e:
                    print(f"Error loading {file}: {e}")

    images = np.array(images)
    labels = to_categorical(np.array(labels))
    return images, labels, class_names

# Load data
X_train, y_train, class_names = load_data("train")
X_test, y_test, _ = load_data("test")

# Normalize
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

# Reshape for CNN input
X_train = X_train.reshape((-1, 64, 64, 1))
X_test = X_test.reshape((-1, 64, 64, 1))

# Shuffle training data (optional but good practice)
shuffle_idx = np.random.permutation(len(X_train))
X_train, y_train = X_train[shuffle_idx], y_train[shuffle_idx]

# Model architecture
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(class_names), activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# Callbacks
callbacks = [
    ModelCheckpoint("hand_sign_model.h5", save_best_only=True),
    EarlyStopping(patience=5, restore_best_weights=True)
]

# Train
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=30,
    batch_size=32,
    callbacks=callbacks
)

# Save class names
with open("class_names.txt", "w") as f:
    f.write("\n".join(class_names))

print("✅ Training completed! Model saved as hand_sign_model.h5")
