from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from torch.utils.data import TensorDataset, DataLoader
import torch
from preprocessing import X_test, X_train, y_test, y_train
from glove_embedding import load_glove
import numpy as np



tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(X_train)
word_index = tokenizer.word_index

X_train_seq = tokenizer.texts_to_sequences(X_train)

X_test_seq = tokenizer.texts_to_sequences(X_test)

max_len = 100

X_train_pad = pad_sequences(X_train_seq, maxlen=max_len)

X_test_pad = pad_sequences(X_test_seq, maxlen=max_len)


X_train_tensor = torch.tensor(X_train_pad)

y_train_tensor = torch.tensor(y_train.values)

X_test_tensor = torch.tensor(X_test_pad)

y_test_tensor = torch.tensor(y_test.values)

train_dataset = TensorDataset(
    X_train_tensor,
    y_train_tensor
)

test_dataset = TensorDataset(
    X_test_tensor,
    y_test_tensor
)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32
)

if __name__ == "__main__":
    print("Dataset prepared")
