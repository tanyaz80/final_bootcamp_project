# Gather the news
# Load vectorizer
# Vectorize
# Load model
# Predict with model

import streamlit as st
import pandas as pd
import numpy as np
import functions
from PIL import Image
#image = Image.open('fakenews.jpg')
#st.image(image)

lr_text = functions.load("../Models/lr.pickle")
lr_title = functions.load("../Models/lr_title.pickle")
pac_text = functions.load("../Models/pac.pickle")
pac_title = functions.load("../Models/pac_title.pickle")
rfc_text = functions.load("../Models/rfc.pickle")
rfc_title = functions.load("../Models/rfc_title.pickle")
vectorization_title = functions.load("../Vectorizers/vectorization_title.pickle")
vectorization_text = functions.load("../Vectorizers/vectorization_text.pickle")


def news_prediction(text_string, is_news=True):
    #news_string = str(input())
    testing_news = {"text":[text_string]} #making a dictionary with the key - "text"
    new_def_test = pd.DataFrame(testing_news) #making a data frame for testing
    new_def_test["text"] = new_def_test["text"].apply(functions.clean_up) #cleaning up
    new_x_test = new_def_test["text"] #taking only text 
    if ( is_news==True ):
        new_xv_test = vectorization_text.transform(new_x_test) #vectorization
        pred_LR = lr_text.predict(new_xv_test) #predictions
        pred_PAC = pac_text.predict(new_xv_test)
        pred_RFC = rfc_text.predict(new_xv_test) 
        if (pred_LR[0]==0)&(pred_PAC[0]==0)&(pred_RFC[0]==0):
            st.error('All models agree: it is a FAKE NEWS')
        elif (pred_LR[0]==1)&(pred_PAC[0]==1)&(pred_RFC[0]==1):
            st.success('All models agree: it is a NOT A FAKE NEWS')
        else: 
            st.text('Models came to different conclusions:')
            st.text('Logistic regression prediction is (0-fake,1-true):'+str(pred_LR[0]))
            st.text('Passive Agressive Classifier prediction is (0-fake,1-true): '+str(pred_PAC[0]))
            st.text('Random Forest Classifier prediction is (0-fake, 1-true)'+str(pred_RFC[0]))
    else:
        new_xv_test = vectorization_title.transform(new_x_test) #vectorization
        pred_LR_title = lr_title.predict(new_xv_test) #predictions
        pred_PAC_title = pac_title.predict(new_xv_test)
        pred_RFC_title = rfc_title.predict(new_xv_test) 
        if (pred_LR_title[0]==0)&(pred_PAC_title[0]==0)&(pred_RFC_title[0]==0):
            st.error('All models agree: it is a FAKE NEWS')
        elif (pred_LR_title[0]==1)&(pred_PAC_title[0]==1)&(pred_RFC_title[0]==1):
            st.success('All models agree: it is a NOT A FAKE NEWS')
        else: 
            st.text('Models came to different conclusions:')
            st.text('Logistic regression prediction is (0-fake,1-true):'+str(pred_LR_title[0]))
            st.text('Passive Agressive Classifier prediction is (0-fake,1-true): '+str(pred_PAC_title[0]))
            st.text('Random Forest Classifier prediction is (0-fake, 1-true)'+str(pred_RFC_title[0]))
        

def main():

    image = Image.open('fakenews.jpeg')
    st.image(image)

    st.header('Fake News Detector App')
    
    options1 = st.selectbox(
    'What would you like to test being fake: text or title',
      ['TEXT', 'TITLE'], key="1")

    if ( options1 == 'TEXT'):
        news = st.text_input('Input a news:')
        if st.button('News Detector Result'):
            news_prediction(news)
    else:
        title = st.text_input('Input a title:')
        if st.button('Title Detector Result'):
            news_prediction(title, is_news=False)

if __name__== "__main__":
    main()





#st.write('Checking on:', news)
