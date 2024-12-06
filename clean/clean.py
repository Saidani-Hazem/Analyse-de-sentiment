import pandas as pd
import re



data = pd.read_csv("abc.csv")


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

data.to_csv('data.csv',index=False)
print(data)
