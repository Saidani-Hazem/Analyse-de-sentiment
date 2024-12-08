import pandas as pd
import re
data = pd.read_csv(r"C:\Users\USER\Desktop\Jupiter\Analyse-de-sentiment\clean\abc.csv")
words = ['and', 'or', 'the', 'is', 'a', 'of', 'to', 'in', 'on', 'for', 'with', 'at', 'by', 'an', 'be', 'this', 'that', 'it']

def clean(texte):
    
    texte = texte.lower()
    
    texte = re.sub(r"http\S+|www\S+|https\S+", '', texte)
    
    texte = re.sub(r'@\w+', '', texte)
    
    texte = re.sub(r'#', '', texte)
    
    texte = re.sub(r"[^a-z\s]", '', texte)
    
    texte = re.sub(r'\s+', ' ', texte).strip()

    texte = ' '.join([word for word in texte.split() if word not in words])

    return texte

data = data.dropna(subset=["text"])
data = data.dropna(subset=["label"])
data = data[data['label'].isin([0,1,2,3,4,5])]

data['text'] = data['text'].apply(clean)
data.drop(columns=data.columns[0],axis=1)
data.to_csv('data.csv',index=False)
X = data["text"]
Y = data["label"]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_test_tfidf)
classificationResults = classification_report(y_test, y_pred,output_dict=True)
accuracyScore = classificationResults["accuracy"]

feelings = ["SADNESS","JOY","LOVE","ANGER","FEAR","SURPRISE"]
def PredictWhatUserTyped(text):
    sampleToTest = vectorizer.transform([text])
    samplePrediction = model.predict(sampleToTest)
    return feelings[samplePrediction[0]]


import streamlit as st

st.header("Data Visualization")


st.write("Le modele utilise est Linear Regression ")
st.write(f"Le score du modele est : {accuracyScore} ")

TextInsertedByUser = st.text_input("type anything u want here !")
if st.button("Click Me!"):
    if TextInsertedByUser:
        result = PredictWhatUserTyped(TextInsertedByUser)
        print(result)
        st.success("Prediction: " + str(result))
    else:
        st.warning("Please type something before clicking the button!")

# Displaying the table
st.write("### Database Table")
st.dataframe(data) 

# Plot 1: Simple Line Chart
st.subheader("Line Chart")
# Footer
st.sidebar.info("Powered by Streamlit")