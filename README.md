# fashion_mnist_project
Made by: Palak Mangla (B.Tech AIML from IGDTUW) 2022-2026 Batch

🧵 Fashion-MNIST CNN Web Application
📌 Overview

This project implements a Convolutional Neural Network (CNN) for classifying images from the Fashion-MNIST dataset into 10 categories (e.g., T-shirt, Dress, Sneakers, etc.).
The model is integrated into a web application that allows users to upload an image and view predictions.

📂 Project Structure
fashion-mnist-cnn-app/
│── models/
│   └── cnn.keras          # Saved trained model
│── train_cnn.py           # Script to train the CNN
│── app.py                 # Web app (Streamlit or Flask/FastAPI)
│── requirements.txt       # Python dependencies
│── README.md              # Project documentation

🚀 Features

Train a CNN on Fashion-MNIST dataset.
Save and load trained models (cnn.keras).
Web app for image upload and prediction.
REST API endpoints (if using Flask/FastAPI).

Deployed on cloud (Vercel / Render / Streamlit Cloud).

⚙️ Installation

Clone the repository:

git clone https://github.com/your-username/fashion-mnist-cnn-app.git
cd fashion-mnist-cnn-app


Create a virtual environment & install dependencies:

python -m venv venv
source venv/bin/activate   # For Linux/Mac
venv\Scripts\activate      # For Windows
pip install -r requirements.txt

🏋️‍♂️ Training the Model

Run the training script:

python train_cnn.py

This will:

Download Fashion-MNIST dataset.

Train a CNN model.

Save the trained model to models/cnn.keras.
