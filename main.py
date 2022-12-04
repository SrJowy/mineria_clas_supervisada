import pandas as pd
from nltk.stem import WordNetLemmatizer, PorterStemmer
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
import string
import gensim
import numpy as np
from Perceptron import SimplePerceptron
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score, roc_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
from matplotlib import pyplot

pData = []
n_topicos = 100
label = pd.DataFrame()
n_instancias = 100

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
    #df = df[:n_instancias]
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

def entrenar_modelo(X_train, X_test, y_train, y_test, depth, cr):
    dTree = RandomForestClassifier(min_samples_split=depth, criterion= cr)

    dTree.fit(X_train, y_train)
    prob = dTree.predict_proba(X_test)
    y_pred = dTree.predict(X_test)
    print('\nRESULTADOS:\n')
    print(f1_score(y_test, y_pred, average='weighted'))
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    return y_pred, prob

if __name__ =="__main__":
    pd.options.mode.chained_assignment = None
    pData = cargar_datos()
    label = pData['label']
    pData = lda().tolist()
    X = pd.DataFrame(pData)
    X = X.drop(X.columns[[0]], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X.values, label.values, test_size=0.3)
    criterion = ["gini", "entropy"]
    
    for i in range(2,11,2):
        for j in range(0,2):
            y_pred, prob = entrenar_modelo(X_train, X_test, y_train, y_test, i, criterion[j])
            lr_probs = prob[:, 1]
            ns_probs = [0 for _ in range(len(y_test))]
            ns_auc = roc_auc_score(y_test, ns_probs)
            lr_auc = roc_auc_score(y_test, lr_probs)
        
            print('Sin entrenar: ROC AUC=%.3f' % (ns_auc))
            print('Tree: ROC AUC=%.3f' % (lr_auc))
        
            ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
            lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)
            
            pyplot.plot(lr_fpr, lr_tpr, marker='.', label='Tree ' + str(i) + " " + criterion[j])
    
    pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='Sin entrenar')
    
    
    pyplot.xlabel('Tasa de Falsos Positivos')
    pyplot.ylabel('Tasa de Verdaderos Positivos')
    pyplot.legend()
    pyplot.show()