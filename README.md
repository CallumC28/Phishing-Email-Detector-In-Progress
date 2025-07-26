# 📨 Phishing Email Detector

A desktop application that detects phishing emails using a fine-tuned BERT model. Features a Tkinter GUI that accepts input via file upload, clipboard paste, or screen OCR. Built with PyTorch and HuggingFace Transformers, this project combines cybersecurity and NLP in a user-friendly tool.

> ⚠️ **Note:** The model is not yet trained - development is in progress and a suitable dataset is currently being sourced for real-world accuracy testing.

---

## Features

- AI-powered phishing detection using BERT (`bert-base-uncased`)
- Input methods: file upload, clipboard paste, or screen OCR
- Real-time model training and validation using HuggingFace Trainer
- Live GUI logs and classification reports after training
- Standalone desktop interface built with Tkinter

---

## 🧠 Tech Stack

| Category         | Technologies                         |
|------------------|--------------------------------------|
| Language         | Python                               |
| NLP & ML         | PyTorch, HuggingFace Transformers    |
| Data Handling    | Pandas, scikit-learn                 |
| GUI              | Tkinter                              |
| OCR              | Tesseract OCR via `pytesseract`      |
| Extras           | Pillow, Rich, Threading              |

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/CallumC28/Phishing-Email-Detector.git
cd phishing-email-detector
```

---

## 🧪 Training the Model
Launch the app:

```bash
python phishing_detector.py
```
In the GUI:

Click Browse CSV to select your dataset

Click Train Model to start training

Training logs and accuracy report will appear in real time

---

## 🔍 Scanning Emails
Use one of the input options in the GUI:

Load File: Import a .txt file

Paste: Paste email content from clipboard

Scan Screen: Capture screen using OCR

Click Scan Email to run detection. The result will indicate whether the email is likely phishing or legitimate, with a confidence score.

---

## 📸 Screenshots

### GUI Demo

![Phishing Email Detector GUI](screenshots/gui.png)

---

## ✅ TODO
 Source and clean a reliable phishing email dataset

 Train and evaluate BERT model on real data

 Add support for email header parsing

 Export scan results as PDF

 Add URL/link spoof detection

