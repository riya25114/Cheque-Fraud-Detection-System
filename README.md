# 📝 Cheque Fraud Detection System  

A machine learning based system for detecting forged cheque signatures using a Siamese Neural Network.  
The model compares the signature on a cheque with a reference signature to verify authenticity.  

---

## 🚀 Features
- Detect forged cheque signatures with deep learning.  
- Flask web interface for uploading cheques & reference signatures.  
- Extendable to other fraud detection tasks (amount tampering, MICR validation, etc.).  

---

## 📂 Project Structure
    ├── app.py # Flask web app
    ├── cheque.ipynb # Model training & experiments
    ├── inspect_model.py # Model inspection / testing
    ├── siamese_signature_model.pth # Trained model weights
    ├── requirements.txt # Python dependencies
    ├── template/ # HTML templates for Flask
    └── README.md # Project documentation
