import streamlit as st  
import pickle
loaded_model = pickle.load(open('News_classification.sav','rb')) 
from sklearn.ensemble import RandomForestClassifier 
import re 
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
nltk.download('punkt') 

vectorize = pickle.load(open('CountVectorize.sav','rb'))
model = pickle.load(open('News_classification.sav','rb'))
 


def preprocess(text):
    remove = re.compile(r'')
    after_removed_tags= re.sub(remove, '', text) 
    removed_special_char = ''
    for x in after_removed_tags:
        if x.isalnum():
            removed_special_char = removed_special_char + x
        else:
            removed_special_char = removed_special_char + ' '
    lowered_text = removed_special_char.lower()  
    
    stop_words = set(stopwords.words('english')) 
    
    words = word_tokenize(lowered_text)
    after_stopwords = [x for x in words if x not in stop_words]  
    
    wordnet = WordNetLemmatizer()
    return " ".join([wordnet.lemmatize(word) for word in after_stopwords]) 


st.title("News Classification") 
input_text = st.text_area('Enter your News article text') 
if st.button('Predict'):

    # 1. preprocess
    prepcoess_text = preprocess(input_text) 
    # 2. vectorize
    vector_input_text = vectorize.transform([prepcoess_text])
    # 3. predict
    output = model.predict(vector_input_text) 
    # 4. Display
    if output == [0]:
        result = "Business News"
    elif output == [1]:
        result = "Tech News"
    elif output == [2]:
        result = "Politics News"
    elif output == [3]:
        result = "Sports News"
    elif output == [1]:
        result = "Entertainment News"
    st.header(result)


