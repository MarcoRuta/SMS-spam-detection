import gradio as gr
import pickle
from sklearn.feature_extraction.text import CountVectorizer

def spam_checker(sms):
    print(sms)

    sms = [sms]

    # load the vectorizer
    loaded_vectorizer = pickle.load(open('trained_models/vectorizer.pickle', 'rb'))

    # load the models
    loaded_model_NB = pickle.load(open('trained_models/Naive Bayes.model', 'rb'))
    loaded_model_SVM = pickle.load(open('trained_models/SGD SVM.model', 'rb'))

    # making a prediction with both the algorithms
    result_NB = loaded_model_NB.predict(loaded_vectorizer.transform(sms))
    result_SVM = loaded_model_SVM.predict(loaded_vectorizer.transform(sms))

    if(result_NB or result_SVM):
         return "Pay attention! This SMS is suspicious and can be SPAM!"
    else: return "This SMS seems legit!"


gr.Interface(fn=spam_checker, inputs=["text"], outputs=["text"]).launch(share = True)
