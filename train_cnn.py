# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import fashion_mnist

# Load Fashion-MNIST dataset
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

# Normalize to [0,1]
X_train = X_train / 255.0
X_test = X_test / 255.0

# Flatten for ML models (28x28 -> 784 features)
X_train_flat = X_train.reshape(len(X_train), -1)
X_test_flat = X_test.reshape(len(X_test), -1)

# Class names for better readability
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


# Reshape for CNN (28,28,1)
X_train_cnn = X_train.reshape(-1, 28, 28, 1)
X_test_cnn = X_test.reshape(-1, 28, 28, 1)

# Build CNN model
cnn = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(10, activation='softmax')
])

cnn.compile(optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

# Train CNN
history = cnn.fit(X_train_cnn, y_train, epochs=10, batch_size=64,
                  validation_split=0.2, verbose=2)

# Evaluate on test set
test_loss, test_acc = cnn.evaluate(X_test_cnn, y_test, verbose=0)
print("CNN Test Accuracy:", test_acc)

# Predictions
y_pred_cnn = np.argmax(cnn.predict(X_test_cnn), axis=-1)

print(classification_report(y_test, y_pred_cnn, target_names=class_names))

# Summary comparison
results = {
    "CNN": test_acc
}

cnn.save("models/fashion_cnn.keras")

print("Model Performance Comparison:")
for model, acc in results.items():
    print(f"{model}: {acc:.4f}")

plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.legend()
plt.title("CNN Accuracy")
plt.show()

