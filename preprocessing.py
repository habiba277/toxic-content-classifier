import pandas as pd
import re
import nltk
from nltk import word_tokenize, sent_tokenize, pos_tag , WordNetLemmatizer
from nltk.corpus import stopwords, wordnet
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler

df = pd.read_excel("data/NLP_Neurova_toxic_content_classification.xlsx")

df["text"] = df["query"] + " " + df["image descriptions"]

df = df.drop_duplicates(subset=['text'])
df = df.reset_index(drop=True)

if __name__ == "__main__":

    print(df.head())
    print(df["Toxic Category"].value_counts())

StopWords = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def get_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('R'):
        return wordnet.ADV
    elif tag.startswith('N'):
        return wordnet.NOUN
    return wordnet.NOUN

def preprocessing_text(text):
    text = text.lower()  
    text = re.sub(r"\s+"," ",text).strip()  # remove extra spaces
    text = re.sub(r'http\S+|www\S+', '', text) # remove urls
    text = re.sub(r'[^\w\s]', '', text)  # remove punctuation
    text = re.sub(r'\d+', '', text)  # remove numbers


    tokens = word_tokenize(text)
    tags = pos_tag(tokens)
    new_tokens = [ lemmatizer.lemmatize(word.lower(),get_pos(tag)) for word,tag in tags 
                   if ( word not in StopWords and len(word) > 2)]

        
    return " ".join(new_tokens)

df["text"] = df["text"].apply(preprocessing_text)

rare_classes = [
    "Suicide & Self-Harm",
    "Elections",
    "Sex-Related Crimes",
    "Child Sexual Exploitation"
]

df["Toxic Category"] = df["Toxic Category"].replace(
    rare_classes,
    "Rare Toxic"
)
if __name__ == "__main__":

    print(df["Toxic Category"].value_counts())


encoder = LabelEncoder()

df["label"] = encoder.fit_transform(df["Toxic Category"])


X_train, X_test, y_train, y_test = train_test_split(
    df["text"],
    df["label"],
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)


