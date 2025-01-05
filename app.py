import streamlit as st
import pickle
import string
import nltk
import nltk.tokenize.punkt
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download('punkt')
ps = PorterStemmer()

tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))


def transform_text(text):
    text = text.lower()  # lowercasing
    text = nltk.word_tokenize(text)  # tokenization
    y = []
    # removing special characters
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]  # cloning
    y.clear()

    # removing punctuation
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        # stemming of words like playing->play
        y.append(ps.stem(i))

    return " ".join(y)


st.title("Emial/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button("Predict"):
    # 1. Preprocess

    transform_sms = transform_text(input_sms)

    # 2. Vectorize

    vector_input = tfidf.transform([transform_sms])
    # 3. Predict
    result = model.predict(vector_input)[0]

    # 4. Display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
