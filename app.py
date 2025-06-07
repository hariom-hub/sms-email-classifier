import streamlit as st
import pickle
import string
import nltk
import nltk.tokenize.punkt
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

import os
# Set the NLTK data path to the local nltk_data folder
nltk.data.path.append(os.path.join(os.getcwd(), 'nltk_data'))

nltk.download('punkt')
ps = PorterStemmer()

tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))


# def transform_text(text):
#     text = text.lower()  # lowercasing
#     text = nltk.word_tokenize(text)  # tokenization
#     y = []
#     # removing special characters
#     for i in text:
#         if i.isalnum():
#             y.append(i)
#
#     text = y[:]  # cloning
#     y.clear()
#
#     # removing punctuation
#     for i in text:
#         if i not in stopwords.words('english') and i not in string.punctuation:
#             y.append(i)
#
#     text = y[:]
#     y.clear()
#
#     for i in text:
#         # stemming of words like playing->play
#         y.append(ps.stem(i))
#
#     return " ".join(y)


# Replace your transform_text function with this more robust version

def transform_text(text):
    try:
        # Convert to lowercase
        text = text.lower()

        # Try NLTK tokenization first, fall back to simple split if fails
        try:
            text = nltk.word_tokenize(text)
        except LookupError:
            # Fallback if NLTK data is missing
            import re
            text = re.findall(r'\b\w+\b', text.lower())
        except Exception:
            # Ultimate fallback
            text = text.split()

        # Remove non-alphanumeric characters
        text = [char for char in text if char.isalnum()]

        # Remove stopwords (with fallback)
        try:
            from nltk.corpus import stopwords
            stop_words = set(stopwords.words('english'))
            text = [char for char in text if char not in stop_words]
        except LookupError:
            # Basic English stopwords fallback
            basic_stopwords = {'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
                               'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
                               'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
                               'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
                               'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
                               'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
                               'while', 'of', 'at', 'by', 'for', 'with', 'through', 'during', 'before', 'after',
                               'above', 'below', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
                               'further', 'then', 'once'}
            text = [char for char in text if char not in basic_stopwords]

        # Stemming with fallback
        try:
            from nltk.stem import PorterStemmer
            ps = PorterStemmer()
            text = [ps.stem(char) for char in text]
        except:
            # Skip stemming if NLTK fails
            pass

        return " ".join(text)

    except Exception as e:
        st.error(f"Error in text processing: {e}")
        return text  # Return original text if all else fails
st.title("Email/SMS Spam Classifier")

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
