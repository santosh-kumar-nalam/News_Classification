import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Ensure NLTK data is downloaded
nltk.data.path.append('./nltk_data')
nltk.download('stopwords', download_dir='./nltk_data')
nltk.download('wordnet', download_dir='./nltk_data')
nltk.download('punkt', download_dir='./nltk_data')

# Load the pre-trained model and vectorizer
loaded_model = pickle.load(open('News_classification.sav', 'rb'))
vectorize = pickle.load(open('CountVectorize.sav', 'rb'))

# Preprocess the input text
def preprocess(text):
    # Remove special characters and tags
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.lower().strip()

    # Tokenize and remove stopwords
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    words = [word for word in words if word not in stop_words]

    # Lemmatize the words
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]

    return ' '.join(words)

# Streamlit UI
st.title("News Classification")
input_text = st.text_area('Enter your News article text')

if st.button('Predict'):
    # 1. Preprocess the input text
    preprocessed_text = preprocess(input_text)

    # 2. Vectorize the preprocessed text
    vectorized_text = vectorize.transform([preprocessed_text])

    # 3. Predict the category
    output = loaded_model.predict(vectorized_text)

    # 4. Display the result
    
    categories = {0: "Business News", 1: "Tech News", 2: "Politics News", 3: "Sports News", 4: "Entertainment News"}
    result = categories.get(output[0], "Unknown Category")
    st.header(result)
