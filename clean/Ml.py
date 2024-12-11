from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import seaborn as sns
import re


#importaion  nettoyage des donnees

data = pd.read_csv("data.csv")

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



X = data["text"]
Y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_test_tfidf)
classificationResults = classification_report(y_test, y_pred,output_dict=True)
accuracyScore = classificationResults["accuracy"]
cm = confusion_matrix(y_test,y_pred)

feelings = ["SADNESS","JOY","LOVE","ANGER","FEAR","SURPRISE"]

def PredictWhatUserTyped(text):
    sampleToTest = vectorizer.transform([text])
    samplePrediction = model.predict(sampleToTest)
    return feelings[samplePrediction[0]]

#interface



st.header("Description des données : ")

st.write("Les données collectées contiennent des commentaires textuels étiquetés par des sentiments de 0 à 5, correspondant à tristesse, joie, amour, colère, peur et surprise. Elles serviront à entraîner un modèle prédictif pour analyser les émotions dans les textes.")
st.write("# Database Table")
st.dataframe(data.head(10)) 

counts = data['label'].value_counts()

fig, ax = plt.subplots(figsize=(5, 5))
ax.pie(counts, labels=counts.index, startangle=90, colors=("purple","pink","gray","orange","hotpink","lightgreen"))
ax.set_title("\n Distribution des sentiments dans les données")
ax.axis('equal')
st.pyplot(fig)

st.write("0: SADNESS  1: JOY 2: LOVE 3: ANGER 4: FEAR 5: SURPRISE")


st.write("# Le modele utilise est Linear Regression")
st.write("La régression linéaire a été choisie pour sa simplicité et sa capacité à modéliser les relations entre les données textuelles et les scores de sentiment.")
score = f"{accuracyScore * 100:6.2f} %" 
st.write("# Le score de modele : " ,score)







st.title("Matrice de Confusion")
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
            xticklabels=feelings,
            yticklabels=feelings)
plt.ylabel('Vraies classes')
plt.xlabel('Classes prédites')
plt.title('Matrice de Confusion')
st.pyplot(fig)
#-------------------------------------------------------
st.title("Pie")

st.pyplot(plt.pie(data=[data["label"]]))






st.write("#Mots les plus représentatifs par sentiment")
feature_names = vectorizer.get_feature_names_out()
top_words = {}

for i, feeling in enumerate(feelings):
    top_indices = model.coef_[i].argsort()[-10:][::-1]
    top_words[feeling] = [feature_names[idx] for idx in top_indices]

for feeling, words in top_words.items():
    st.write(f"### {feeling}:")
    st.write(", ".join(words))


st.title("Performances par Sentiment")
metrics = pd.DataFrame(classificationResults).T.iloc[:-3, :3]
metrics.rename(columns={"precision": "Précision", "recall": "Rappel", "f1-score": "F1-Score"}, inplace=True)

fig, ax = plt.subplots(figsize=(10, 5))
metrics.plot(kind="bar", ax=ax)
plt.title("Performances du Modèle par Sentiment")
plt.ylabel("Score")
plt.xticks(rotation=45)
st.pyplot(fig)


st.title("Tester Le Modele :")
TextInsertedByUser = st.text_input("type anything u want here !")
if st.button("Click Me!"):
    if TextInsertedByUser:
        result = PredictWhatUserTyped(TextInsertedByUser)
        print(result)
        st.success("Prediction: " + str(result))
    else:
        st.warning("Please type something before clicking the button!")


