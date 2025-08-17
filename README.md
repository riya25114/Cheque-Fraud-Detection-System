# 🏦 Cheque Fraud Detection System

This project is a **deep learning–based Cheque Fraud Detection System** that identifies forged cheques by analyzing **handwritten signatures**.  
It uses a **Siamese Neural Network (SNN)** trained on signature datasets to verify whether the signature on a cheque is genuine or forged.  

The system provides a **web interface (Flask + HTML)** where users can upload cheque/signature images and get instant predictions.

---

## 🚀 Features
- Detects **forged vs. genuine signatures** on cheques.
- Built using **PyTorch** (for Siamese Network training).
- Deployed with **Flask backend + HTML frontend**.
- Supports **real-time user uploads** of cheque/signature images.
- Dataset flexibility (works with multiple signature datasets).

---

## 📂 Project Structure
    cheque_project/
        ── app.py
        ── siamese_signature_model.pth # Trained Siamese model (PyTorch)
        ── template/
            └── index.html # Frontend upload form
            └── result.html # Frontend upload form
        ── inspect_model.py # Strcucture of the model
        ── cheque.ipynb # Model Code 
        ── venv
        ── requirements.txt # Project dependencies
        ── README.md # Project documentation

---

## ⚙️ Installation & Setup

### 1️⃣ Clone Repository
```bash
git clone https://github.com/yourusername/cheque-fraud-detection.git
cd cheque-fraud-detection
### 2️⃣ Create Virtual Environment
python3 -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Run Flask App
python app.py


Visit 👉 http://127.0.0.1:5000/

🔮 Future Work

Extend fraud detection beyond signatures (amount tampering, MICR line validation, etc.).

Integrate OCR (EasyOCR / Tesseract) to extract cheque fields.

Deploy as a full web app with user authentication.

Train with larger signature datasets for higher accuracy.

📌 Credits

Siamese Network implementation inspired by signature verification research papers.

Dataset sources: GPDS, CEDAR, and GitHub open datasets.

Developed by Riya Garg.