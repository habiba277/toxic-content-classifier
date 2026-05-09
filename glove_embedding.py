import numpy as np

def load_glove(path, word_index, embed_dim=100):

    embeddings_index = {}

    with open(path, encoding="utf8") as f:

        for line in f:

            values = line.split()

            word = values[0]

            vector = np.asarray(values[1:], dtype='float32')

            embeddings_index[word] = vector

    vocab_size = len(word_index) + 1

    embedding_matrix = np.zeros((vocab_size, embed_dim))

    for word, i in word_index.items():

        vector = embeddings_index.get(word)

        if vector is not None:

            embedding_matrix[i] = vector

    return embedding_matrix