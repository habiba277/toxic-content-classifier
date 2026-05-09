# Toxic Content Classification using BiLSTM

This project is a complete NLP toxic content classification system built using PyTorch and BiLSTM architecture with GloVe word embeddings.

The model classifies toxic text into multiple toxicity categories using deep learning techniques.

---

# Project Overview

The project includes:

- Data preprocessing
- Text cleaning
- Tokenization
- Lemmatization
- Handling imbalanced data
- GloVe embeddings
- BiLSTM model
- Training pipeline
- Evaluation metrics
- Inference system

---

# Technologies Used

- Python
- PyTorch
- NLTK
- Scikit-learn
- TensorFlow/Keras (Tokenizer & Padding only)
- Pandas
- NumPy
- Matplotlib
- Seaborn

---

# Dataset

Dataset used for this project:

Toxic Content Classification Dataset

The dataset contains:
- User queries
- Image descriptions
- Toxicity labels

---

# Project Structure

```text
project/
│
├── data/
│   └── NLP_Neurova_toxic_content_classification.xlsx
│
├── models/
│   ├── glove.6B.100d.txt
│   ├── toxic_lstm.pth
│   ├── tokenizer.pkl
│   └── label_encoder.pkl
│
├── preprocessing.py
├── glove_embedding.py
├── dataset.py
├── model.py
├── train.py
├── inference.py
├── main.py
│
├── requirements.txt
├── README.md
└── notebook.ipynb
```

# Data Preprocessing 

The preprocessing pipeline includes:

- Lowercasing text
- Removing URLs
- Removing punctuation
- Removing numbers
- Tokenization
- Stopword removal
- POS tagging
- Lemmatization
- Duplicate removal

# Handling Imbalanced Data

The project handles class imbalance using:

- Random Oversampling
- Class Weights

- Rare classes are merged into a single category to improve model generalization.

# Model Architecture

The model is built using:

- Embedding Layer (GloVe pretrained embeddings)
- Bidirectional LSTM
- Dropout Layer
- Fully Connected Layer

# Download GloVe Embeddings

Download GloVe embeddings from:

https://nlp.stanford.edu/projects/glove/

# Training

To train the model:

```bash
python train.py
```

The training pipeline:

- Loads and preprocesses data
- Builds vocabulary
- Loads GloVe embeddings
- Trains the BiLSTM model
- Evaluates performance
- Saves trained weights

# Inference

To run inference:

```bash
python main.py
```

Example:

```text
Enter text: I hate you and want to hurt you
Prediction: Violent Crimes
```

# Evaluation Metrics

The model is evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

# Features

- End-to-end NLP pipeline
- Deep learning based classification
- GloVe embeddings support
- Reusable inference system
- Modular project structure

# Future Improvements

Possible future improvements:

Attention Mechanism
Transformer Models (BERT)
FastText Embeddings
Better imbalance handling
Hyperparameter tuning
Streamlit deployment


# Author

Habiba Ayman

Biomedical Engineering Graduate
Interested in Artificial Intelligence, NLP, and Computer Vision