import json, random, nltk, pickle
import numpy as np
import tensorflow as tf
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

nltk.download('punkt')
stemmer = PorterStemmer()

with open("chatbot/intents.json") as file:
    data = json.load(file)

corpus, tags = [], []
for intent in data["intents"]:
    for pattern in intent["patterns"]:
        tokens = nltk.word_tokenize(pattern)
        stemmed = [stemmer.stem(w.lower()) for w in tokens]
        corpus.append(" ".join(stemmed))
        tags.append(intent["tag"])

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus).toarray()

encoder = LabelEncoder()
y = encoder.fit_transform(tags)

model = Sequential()
model.add(Dense(128, input_shape=(len(X[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dense(len(set(tags)), activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, epochs=200, batch_size=8)

model.save("chatbot/chatbot_model.h5")
pickle.dump(vectorizer, open("chatbot/vectorizer.pkl", "wb"))
pickle.dump(encoder, open("chatbot/encoder.pkl", "wb"))