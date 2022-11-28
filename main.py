import pandas as pd
from nltk.stem import WordNetLemmatizer, PorterStemmer
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
import string
import gensim
import numpy as np
from Perceptron import SimplePerceptron
from sklearn.metrics import confusion_matrix

pData = []
n_topicos = 100
label = pd.DataFrame()
n_instancias = 10

def preprocesar(c):
    wnl = WordNetLemmatizer()
    ps = PorterStemmer()
    remove=[]
    for i in range(len(c)):
        s = str(c[i])
        if not bool(s):
            remove.append(i)
        else:
            s = s.lower()
            s = s.translate(str.maketrans('','', string.punctuation))
            s = s.translate(str.maketrans('','', string.digits))
            s = s.split()
            stop_words=set(stopwords.words('english'))
            filtered_sentence = []
            for w in s:
                if w not in stop_words and len(w)>1 and w.isascii():
                    w = wnl.lemmatize(w)
                    w = ps.stem(w)
                    if len(w)>1:
                        filtered_sentence.append(w)
            if not bool(filtered_sentence):
                remove.append(i)  
            c[i]=filtered_sentence
    return(c, remove)   

def cargar_datos():
    df = pd.read_csv('datos/train.csv')
    df = df[:n_instancias]
    global label
    label = df[['label']]
    df=df.drop(columns=['title', 'id', 'author'])
    df['text'], eliminar=preprocesar(df['text'])
    eliminar=list(dict.fromkeys(eliminar))
    df=df.drop(df.index[eliminar])
    return(df)

def lda():
    id2word = gensim.corpora.Dictionary(pData['text'])
    id2word.save("id2word")
    corpus = [id2word.doc2bow(doc) for doc in pData['text']]
    lda_model = gensim.models.LdaMulticore(corpus=corpus, id2word=id2word, num_topics=n_topicos, random_state=4)
    doc_lda = lda_model[corpus]
    lda_model.save("lda_model")

    matrix = np.zeros((len(pData), n_topicos+1))
    i = 0
    for t in doc_lda:
        matrix[i][0] = int(i+1)
        for e in t:
            matrix[i][e[0]+1] = e[1]
        i+=1
    return(matrix)  

if __name__ =="__main__":
    pData = cargar_datos()
    pData = lda().tolist()
    X = pd.DataFrame(pData)
    X_train, X_test, y_train, y_test = train_test_split(X.values, label.values, test_size=0.3)
    perceptron = SimplePerceptron()

    perceptron.fit(X_train, y_train)
    y_pred = perceptron.predict(X_test)
    print(confusion_matrix(y_test, y_pred))