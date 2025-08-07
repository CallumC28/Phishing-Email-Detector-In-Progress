# Phishing Email Detector

A desktop application that detects phishing emails using a fine-tuned BERT model. Features a Tkinter GUI that accepts input via file upload, clipboard paste, or screen OCR. Built with PyTorch and HuggingFace Transformers, this project combines cybersecurity and NLP in a user-friendly tool. You will have to train the model youself since github limits the amount of files that can be uploaded.

**Can now be trained with eaither CPU only or GPU acceleration for those with Nvidia GPUs, GPU is recommended since CPU training times are extrememly long even for small datasets such as the current one which would of took around 10 hours but with GPU acceleration it took 5 minutes**

---

## Features

- AI-powered phishing detection using BERT (`bert-base-uncased`)
- Input methods: file upload, clipboard paste, or screen OCR
- Real-time model training and validation using HuggingFace Trainer
- Live GUI logs and classification reports after training
- Standalone desktop interface built with Tkinter
- Automatic GPU detection and acceleration when available **(Only Nvidia GPUs)**
- **Screenshot works but is not accurate at determining phishing or legitimate so avoid for now**

---

## Tech Stack

| Category         | Technologies                         |
|------------------|--------------------------------------|
| Language         | Python                               |
| NLP & ML         | PyTorch, HuggingFace Transformers    |
| Data Handling    | Pandas, scikit-learn                 |
| GUI              | Tkinter                              |
| OCR              | Tesseract OCR via `pytesseract`      |
| Extras           | Pillow, Rich, Threading              |

---

## Prerequisites

- **Python ≥ 3.9**
- **CUDA-capable GPU (optional but recommended for faster training)**
- **CUDA Toolkit** matching your GPU driver (e.g. CUDA 11.8 or 13.0)
- **NVIDIA Drivers** installed and use `nvidia-smi` in cmd to see drivers are installed and working
- **Tesseract OCR** installed and on `PATH` for screen scanning

Install Python dependencies:
**pip installs in requirements.txt**

For GPU accelaration **(For Nvidia GPUs Only)**
```bash
#For CUDA 11.8
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118

#For CUDA 12.1
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu121

#For CUDA 12.4
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu124
```
---

## Training the Model
Launch the app:

```bash
python phishing_detector.py
```
In the GUI:

Click Browse CSV to select your dataset

Click Train Model to start training

Training logs and accuracy report will appear in real time

---

## Scanning Emails
Use one of the input options in the GUI:

- Load File: Import a .txt file

- Paste: Paste email content from clipboard

- Scan Screen: Capture screen using OCR

Click Scan Email to run detection. The result will indicate whether the email is likely phishing or legitimate, with a confidence score (0 to 1.00 with 1.00 being 100% confident).

---

## Screenshots

### GUI Demo

![Phishing Email Detector GUI](screenshot/gui.png)

---

## TODO
 Source and clean a more extensive phishing email dataset

 Train and evaluate BERT model on real data

 Add support for email header parsing

 Export scan results as PDF

 Add URL/link spoof detection


