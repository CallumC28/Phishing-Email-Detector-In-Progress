import os
import threading
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
import torch
from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
import tkinter as tk
from tkinter import filedialog, scrolledtext, messagebox, ttk
from PIL import ImageGrab
import pytesseract
from rich.console import Console

# Suppress TensorFlow logs and disable oneDNN optimisations for cleaner output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Error only
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# Disable TensorFlow integrations in Transformers
os.environ['TRANSFORMERS_NO_TF'] = '1'

console = Console()

# Configuration
MODEL_DIR = 'saved_model'

# Data loading
def load_data(path: str):
    df = pd.read_csv(path)
    # normalise text column
    if 'text' not in df.columns:
        if 'Email Text' in df.columns:
            df = df.rename(columns={'Email Text': 'text'})
        elif 'text_combined' in df.columns:
            df = df.rename(columns={'text_combined': 'text'})
        else:
            raise KeyError("Could not find a text column in CSV.")
    # normalise label column
    if 'label' not in df.columns:
        if 'Email Type' in df.columns:
            df['label'] = df['Email Type'].map({'Phishing Email': 1, 'Safe Email': 0})
        elif 'Label' in df.columns:
            df = df.rename(columns={'Label': 'label'})
        else:
            raise KeyError("Could not find a label column in CSV.")
    # remove empty or null texts
    df = df.dropna(subset=['text', 'label'])
    df = df[df['text'].str.strip().astype(bool)]
    return df['text'].tolist(), df['label'].tolist()

# Tokenizer and dynamic padding collator
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

class PhishingDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item
    def __len__(self):
        return len(self.labels)

# Metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='binary', pos_label=1
    )
    return {
        'accuracy': (preds == labels).mean(),
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

# Training function
def train_model(csv_path: str, log_widget=None):
    
    # Debug for trying to see if its using the GPU
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA device count:", torch.cuda.device_count())
        print("Current CUDA device:", torch.cuda.current_device())
        print("Device name:", torch.cuda.get_device_name(0))
        
    def log(msg, style='green'):
        if log_widget:
            log_widget.insert(tk.END, msg + "\n")
            log_widget.see(tk.END)
        else:
            console.print(msg, style=style)

    log(f"Loading data from {csv_path}...")
    texts, labels = load_data(csv_path)
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    # tokenize and pad dynamically
    train_enc = tokenizer(train_texts, truncation=True, padding=False)
    val_enc = tokenizer(val_texts, truncation=True, padding=False)
    train_ds = PhishingDataset(train_enc, train_labels)
    val_ds = PhishingDataset(val_enc, val_labels)

    log("Initialising model...")
    if os.path.isdir(MODEL_DIR) and os.listdir(MODEL_DIR):
        model = BertForSequenceClassification.from_pretrained(MODEL_DIR, num_labels=2)
    else:
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

    args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=2,                # fewer epochs for speed
        per_device_train_batch_size=16,    # larger batch on GPU
        per_device_eval_batch_size=32,
        logging_dir='./logs',
        learning_rate=2e-5,
        fp16=True,                         # mixed precision on GPU
        gradient_accumulation_steps=1,
        dataloader_num_workers=4,
        do_eval=True
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    log("Starting training...")
    trainer.train()
    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)

    log("Evaluating...")
    preds = trainer.predict(val_ds)
    y_pred = preds.predictions.argmax(axis=1)
    acc = accuracy_score(val_labels, y_pred)
    report = classification_report(val_labels, y_pred, target_names=['legitimate','phishing'])
    log(f"Validation accuracy: {acc:.4f}")
    log(report)
    log("Training complete!")

# Inference function
def is_phishing(text: str):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BertForSequenceClassification.from_pretrained(MODEL_DIR).to(device)
    enc = tokenizer(text, truncation=True, padding=True, return_tensors='pt')
    enc = {k: v.to(device) for k, v in enc.items()}
    out = model(**enc)
    probs = torch.softmax(out.logits, dim=1)[0]
    return probs[1].item(), probs[0].item()

# GUI 
class PhishingApp:
    def __init__(self, root):
        self.root = root
        container = ttk.Frame(root, padding=10, style='TFrame')
        container.pack(fill=tk.BOTH, expand=True)

        # Train section
        ttk.Label(container, text="Train Model", font=('Segoe UI',14,'bold'), style='TLabel').grid(row=0,column=0,columnspan=3,sticky='w')
        self.csv_entry = ttk.Entry(container, width=60)
        self.csv_entry.grid(row=1,column=0,sticky='ew', pady=5)
        ttk.Button(container, text="Browse CSV", command=self.browse_csv).grid(row=1,column=1,padx=5)
        ttk.Button(container, text="Train Model", command=self.train_clicked, style='Success.TButton').grid(row=1,column=2)
        self.log_widget = scrolledtext.ScrolledText(container, height=8, bd=0, relief='flat')
        self.log_widget.grid(row=2,column=0,columnspan=3,sticky='nsew', pady=5)

        # Divider
        ttk.Separator(container, orient='horizontal').grid(row=3,column=0,columnspan=3,sticky='ew', pady=10)

        # Scan section
        ttk.Label(container, text="Scan Email", font=('Segoe UI',14,'bold'), style='TLabel').grid(row=4,column=0,sticky='w')
        toolbar = ttk.Frame(container, style='TFrame')
        toolbar.grid(row=5,column=0,columnspan=3, sticky='w')
        ttk.Button(toolbar, text="Load File", command=self.load_email_file).pack(side=tk.LEFT)
        ttk.Button(toolbar, text="Paste", command=self.paste_clipboard).pack(side=tk.LEFT,padx=5)
        ttk.Button(toolbar, text="Scan Screen", command=self.scan_screen).pack(side=tk.LEFT,padx=5)
        self.email_text = scrolledtext.ScrolledText(container, height=8, bd=0, relief='flat')
        self.email_text.grid(row=6,column=0,columnspan=3,sticky='nsew', pady=5)
        ttk.Button(container, text="Scan Email", command=self.scan_clicked, style='Danger.TButton').grid(row=7,column=0,columnspan=3,pady=5)
        self.result_label = ttk.Label(container, text="Ready", font=('Segoe UI',12), style='TLabel')
        self.result_label.grid(row=8,column=0,columnspan=3,pady=5)

        container.columnconfigure(0, weight=1)
        container.columnconfigure(1, weight=0)
        container.columnconfigure(2, weight=0)
        container.rowconfigure(6, weight=1)

    def browse_csv(self):
        path = filedialog.askopenfilename(filetypes=[("CSV files","*.csv")])
        if path:
            self.csv_entry.delete(0, tk.END)
            self.csv_entry.insert(0, path)

    def train_clicked(self):
        path = self.csv_entry.get().strip()
        if not os.path.exists(path):
            messagebox.showerror("Error","CSV file not found.")
            return
        self.log_widget.delete('1.0',tk.END)
        threading.Thread(target=train_model,args=(path,self.log_widget),daemon=True).start()

    def load_email_file(self):
        path = filedialog.askopenfilename(filetypes=[("Text files","*.txt"),("All files","*.*")])
        if path:
            try:
                text = open(path,'r',encoding='utf-8',errors='ignore').read()
                self.email_text.delete('1.0',tk.END)
                self.email_text.insert('1.0',text)
            except Exception as e:
                messagebox.showerror("Error",f"Failed to load file: {e}")

    def paste_clipboard(self):
        try:
            text = self.root.clipboard_get()
            self.email_text.delete('1.0',tk.END)
            self.email_text.insert('1.0',text)
        except tk.TclError:
            messagebox.showerror("Error","No text on clipboard")

    def scan_screen(self):
        self.root.withdraw()
        time.sleep(0.5)
        img = ImageGrab.grab()
        self.root.deiconify()
        try:
            text = pytesseract.image_to_string(img)
        except pytesseract.pytesseract.TesseractNotFoundError:
            messagebox.showerror(
                "OCR Error",
                (
                    "Tesseract-OCR is not installed or not found in PATH.\n"
                    "Please install it and ensure it's in your system PATH."
                )
            )
            return
        self.email_text.delete('1.0',tk.END)
        self.email_text.insert('1.0',text)
        messagebox.showinfo("Scan Complete","Screen capture successful. Performing phishing scan...")
        self.scan_clicked()

    def scan_clicked(self):
        email = self.email_text.get('1.0',tk.END).strip()
        if not email:
            messagebox.showwarning("Input needed","Provide email via file, clipboard, or screen scan.")
            return
        ps, ls = is_phishing(email)
        if ps>=ls:
            result=f"⚠️  Likely phishing (score: {ps:.2f})"
        else:
            result=f"✅  Legitimate (score: {ls:.2f})"
        self.result_label.config(text=result)
        messagebox.showinfo("Scan Result",result)

# Main entry
def main():
    root = tk.Tk()
    root.title("Phishing Email Detector")
    root.geometry("900x650")
    root.configure(bg='#2e3440')
    style = ttk.Style(root)
    style.theme_use('clam')
    style.configure('TFrame',background='#2e3440')
    style.configure('TLabel',background='#2e3440',foreground='#eceff4')
    style.configure('TButton',background='#5e81ac',foreground='#ffffff')
    style.configure('Success.TButton',background='#4caf50')
    style.configure('Danger.TButton',background='#f44336')
    header=tk.Label(root,text="📨 Phishing Email Detector",font=('Segoe UI',20,'bold'),bg='#5e81ac',fg='#eceff4')
    header.pack(fill=tk.X)
    PhishingApp(root)
    root.mainloop()

if __name__=='__main__':
    main()

#to run - python phishing_detector.py