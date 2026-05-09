from preprocessing import X_train, y_train, encoder
from dataset import train_loader, test_loader, tokenizer
from glove_embedding import load_glove
from model import ToxicLSTM
import torch
import torch.nn as nn
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def train_model():
    embedding_matrix = load_glove(
    "models/glove.6B.100d.txt",
    tokenizer.word_index,
    embed_dim=100
)
    model = ToxicLSTM(
    embedding_matrix=torch.tensor(embedding_matrix, dtype=torch.float),
    hidden_dim=64,
    output_dim=len(encoder.classes_)
)

    weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )

    weights = torch.tensor(weights, dtype=torch.float)

    criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )

    epochs = 20
    best_loss = float('inf')

    for epoch in range(epochs):

        model.train()

        total_loss = 0
        all_preds = []
        all_labels = []

        for inputs, labels in train_loader:

            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())

        acc = accuracy_score(all_labels, all_preds)
        print(f"Epoch {epoch+1}")
        print(f"Loss: {total_loss}")
        print(f"Accuracy: {acc:.4f}")
        # save best model only
        if total_loss < best_loss:

            best_loss = total_loss

            torch.save(
                model.state_dict(),
                "models/toxic_lstm.pth"
            )

            print("Best model saved")
    # evaluation
    model.eval()

    predictions = []
    true_labels = []

    with torch.no_grad():

        for inputs, labels in test_loader:

            outputs = model(inputs)

            preds = torch.argmax(outputs, dim=1)

            predictions.extend(preds.numpy())

            true_labels.extend(labels.numpy())

    print(classification_report(
        true_labels,
        predictions
    ))
    cm = confusion_matrix(true_labels, predictions)

    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d')

    plt.xlabel("Predicted")
    plt.ylabel("True")

    plt.title("Confusion Matrix")

    plt.show()


    # save tokenizer
    joblib.dump(
        tokenizer,
        "models/tokenizer.pkl"
    )

    # save encoder
    joblib.dump(
        encoder,
        "models/label_encoder.pkl"
    )

    print("Everything saved successfully")


if __name__ == "__main__":

    train_model()
