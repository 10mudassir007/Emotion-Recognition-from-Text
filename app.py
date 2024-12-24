import os
import pickle
import numpy as np
import warnings
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

warnings.filterwarnings('ignore')

app = FastAPI()

templates = Jinja2Templates(directory='templates')

class RecognizeEmotion(BaseModel):
    text: str

@app.get('/',response_class=HTMLResponse)
async def home(request:Request):
    return templates.TemplateResponse('index.html',{'request':request})

@app.post('/predict/')
async def predict(txt:RecognizeEmotion):
    with open('tokenizer.pkl','rb') as f:
            tokenizer = pickle.load(f)

    model = load_model('emotion_model.keras')
    
    def preprocess_texts(txts):
        preprocessed_texts = []
        stop_words = set(stopwords.words('english'))
        for txt in txts:
            words = word_tokenize(txt)
            filtered_txt = [word for word in words if word.lower() not in stop_words and word not in string.punctuation]
            txt = ' '.join(filtered_txt)
            preprocessed_texts.append(txt)
        preprocessed_texts = np.array(preprocessed_texts)
        return preprocessed_texts

    def predict_emotion(txt):
        txt = preprocess_texts([txt])
        txt1 = tokenizer.texts_to_sequences(txt)
        txt1 = pad_sequences(txt1,maxlen=35)
        pred = model.predict(txt1)
        
        if np.max(pred) < 0.5:
            return "Undetermined"
        
        pred = np.argmax(pred,axis=1)
        emotions = ['Anger','Fear','Joy','Love','Sadness','Surprise']
        return emotions[pred[0]]
    
    return {'prediction':predict_emotion(txt.text)}
