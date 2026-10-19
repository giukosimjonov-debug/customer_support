# 🏦 Customer Support Ticket Classifier (NLP & Deep Learning)

## Overview
An end-to-end Natural Language Processing (NLP) pipeline built to automatically classify customer support tickets into 77 distinct banking categories. The model processes raw customer messages and predicts the exact intent, which can heavily optimize backend customer support routing.

## Tech Stack
*   **Deep Learning Framework:** PyTorch
*   **Machine Learning & NLP:** Scikit-learn (TF-IDF), NLTK (Lemmatization, Stopwords)
*   **Data Manipulation:** Pandas, Numpy
*   **Visualization:** Matplotlib

## Project Pipeline
1.  **Text Preprocessing:** Lowercasing, punctuation removal, HTML tag stripping, and lemmatization using NLTK.
2.  **Vectorization:** Transformed cleaned text into numerical format using `TfidfVectorizer` (1000 features).
3.  **Neural Network Architecture:** Built a Multi-layer Perceptron (MLP) in PyTorch with 2 hidden layers (512 & 256 neurons) and ReLU activations.
4.  **Training & Evaluation:** Trained using Adam Optimizer and CrossEntropyLoss. Achieved **84.68% accuracy** on unseen test data.

## Future Plans
I plan to integrate this PyTorch model into a backend API using Flask/FastAPI to simulate a real-world automated ticket routing system.
