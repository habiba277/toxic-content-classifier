import torch
import joblib

from tensorflow.keras.preprocessing.sequence import pad_sequences
from glove_embedding import load_glove
from model import ToxicLSTM
from preprocessing import preprocessing_text

# load tokenizer
tokenizer = joblib.load(
    "models/tokenizer.pkl"
)

# load encoder
encoder = joblib.load(
    "models/label_encoder.pkl"
)

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

# load weights
model.load_state_dict(
    torch.load("models/toxic_lstm.pth")
)

model.eval()

def predict(text):

    text = preprocessing_text(text)

    seq = tokenizer.texts_to_sequences([text])

    padded = pad_sequences(seq, maxlen=100)

    tensor = torch.tensor(padded)

    output = model(tensor)

    pred = torch.argmax(output, dim=1)

    return encoder.inverse_transform(
        [pred.item()]
    )[0]

if __name__ == "__main__":

    sample_text = "I hate you and I want to hurt you"

    prediction = predict(sample_text)

    print(f"Text: {sample_text}")
    print(f"Predicted Toxic Category: {prediction}")