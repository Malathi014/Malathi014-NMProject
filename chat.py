import json, random, pickle, nltk
import numpy as np
from nltk.stem import PorterStemmer
from tensorflow.keras.models import load_model

stemmer = PorterStemmer()
model = load_model("chatbot/chatbot_model.h5")
intents = json.load(open("chatbot/intents.json"))
vectorizer = pickle.load(open("chatbot/vectorizer.pkl", "rb"))
encoder = pickle.load(open("chatbot/encoder.pkl", "rb"))

def clean_input(text):
    tokens = nltk.word_tokenize(text)
    stemmed = [stemmer.stem(w.lower()) for w in tokens]
    return " ".join(stemmed)

def get_response(msg):
    cleaned = clean_input(msg)
    X = vectorizer.transform([cleaned]).toarray()
    prediction = model.predict(X)[0]
    tag = encoder.inverse_transform([np.argmax(prediction)])[0]

    for intent in intents["intents"]:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])
    return "Sorry, I didn't understand that."