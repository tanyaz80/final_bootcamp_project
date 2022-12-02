FAKE NEWS DETECTOR
Final Bootcamp Project. December 2022.
Done by Tanya Zanina.

This project was created by using data from here: <https://www.kaggle.com/datasets/jainpooja/fake-news-detection>

The data set has 2 zipped files of true and fake news: 21,417 true and  23,481 fake news.

Project contains:

1) checking the data set on NaN and empty elements.
2) removing the not relevant symbols
3) train-test split
4) applying the vectoization - TFIDF
5) building 3 models: Logistic Regression, Passive Agressive Classifier and Random Forest Classifier.
6) Testing and making validation.
7) The best result was shown by Passive Agressive Classifier.
8) Checking if the titles also can be enough to classify the news - fake or true.
The results were interesting. The accuracy and kappa was smaller than for text, but still can be used.
9) Checking the importance features of the models, to see which word were giving the model most information. I did it for all 3 models, the words that the models have used were not the same.
10) Building WordClouds by using wordcloud library for true and fake news.
11) Building sentiment analysis using nltk.NaiveBayesClassifier. Here there were some obsticles by working with big data set, so the data set has to be devided into chucnks. The words that were shown need more analysis. It's interesting to continue exploring.  For example, number one word was "pic" - it was met 1070 times more in fake news than in true. I can't give an answer to this question now. The code for this analysis you can find here: sentiment_analysis.ipynb

12) Building an app using streamlit to test the models. It's available to do for text and title.
The results are given for all 3 models. You can find the code for the app in folder App/fake_detector_app.py

Conclusions:
Fake News Detector is sensitive to type of news. Depending on the way you cleaned the words, we can get different outcome. The validation was good for given set. As next step - would be interesting to continue sentiment analysis.
