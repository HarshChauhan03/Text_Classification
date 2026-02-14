# 🧠 Deep Learning Text Classification using LSTM | NLP Project

![Python](https://img.shields.io/badge/Python-3.x-blue)
![NLP](https://img.shields.io/badge/Field-NLP-green)
![Deep Learning](https://img.shields.io/badge/Model-LSTM-orange)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

This project demonstrates **Text Classification using Deep Learning (LSTM)** in Natural Language Processing.

Unlike traditional NLP methods (BoW, TF-IDF + ML), this project uses an **Embedding Layer + LSTM Neural Network** to classify text sentiment.

---

## 📌 Project Overview

Text Classification is one of the most important NLP tasks used in:

- Sentiment Analysis  
- Spam Detection  
- News Classification  
- Review Analysis  
- Customer Feedback Systems  

In this project, we classify text into:

- **Positive (1)**
- **Negative (0)**

---

## 🎯 Objective

The goal of this project is to:

✅ Convert raw text into numerical sequences  
✅ Use an Embedding layer to learn word representations  
✅ Apply LSTM (Long Short-Term Memory) network  
✅ Train a deep learning model for sentiment classification  

---

## 🧠 Model Architecture

Text Input
↓
Tokenization
↓
Padding Sequences
↓
Embedding Layer
↓
LSTM Layer
↓
Dense Output Layer (Sigmoid)
↓
Sentiment Prediction


---

## 📂 Project Structure

Day10_DL_Text_Classification/
├── dl_text_classifier.py
└── README.md


---

## ⚙️ Technologies Used

- Python 🐍  
- TensorFlow / Keras  
- NumPy  

---

## 🧠 Model Details

- Embedding Layer: Converts words into dense vectors  
- LSTM Layer: Captures sequential dependencies in text  
- Dense Layer (Sigmoid): Outputs probability for binary classification  

Loss Function: `binary_crossentropy`  
Optimizer: `adam`

---

## ▶️ How to Run

### Step 1 — Install dependencies
```bash
pip install tensorflow numpy
Step 2 — Run the script
python dl_text_classifier.py
✅ Output
Model summary

Training accuracy

Prediction for test sentence

Sentiment output (Positive / Negative)

🚀 Learning Outcomes
By completing this project, you will:

✔ Understand deep learning in NLP
✔ Learn how LSTM works for sequence data
✔ Implement embedding layers
✔ Build a neural network for text classification
✔ Move from traditional NLP → deep NLP

📖 Why This Matters
Deep learning models like LSTM are used in:

Sentiment analysis systems

Chatbots

Voice assistants

Content moderation tools

Recommendation systems

This project builds the foundation for advanced models like BERT and Transformers.

👨‍💻 Author
Harsh Chauhan
Computer Engineering Student
Interested in AI, NLP & Data Science
