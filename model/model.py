#from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn import svm

import pickle


# open a file, where you stored the pickled data
file = open('senate2020.pkl', 'rb')

# dump information to that file
data = pickle.load(file)

df = data.reset_index()
df = df.drop(['level_0'], axis=1)
df = df.set_index('level_1').rename_axis('index')

df_trasponate = df.T


df_trasponate = df_trasponate.fillna(0)


for column in df_trasponate:
  df_trasponate[column].replace({"Si": 1, "No": 2, "Abstencion": 3, "Pareo": 4}, inplace=True)

nuevo2 = df_trasponate



nuevo2 = nuevo2.filter(items=['Latorre R., Juan Ignacio', 'ABSTENCION', 'ETAPA','FECHA', 'NO', 'PAREO', 'QUORUM', 'SESION', 'SI', 'TEMA','TIPOVOTACION'])




for column in nuevo2:
  nuevo2[column].replace({"Discusión particula": 1,"Discusión general":2,"Discusión particular":3}, inplace=True)

for column in nuevo2:
  nuevo2[column].replace({"Primer trámite constitucional": 1,"Segundo trámite constitucional":2,"Tercer trámite constitucional":3}, inplace=True)

for column in nuevo2:
  nuevo2[column].replace({"Disc. informe C.Mixta por rechazo de modific. en C. Origen": 4,"Disc. informe C.Mixta por rechazo de modif. C. Revisora":5,"Disc. Informe C.Mixta por rechazo idea de legislar C. Revis.":6,"Disc. informe C.Mixta por rechazo idea de legislar C. Origen":7}, inplace=True)



for column in nuevo2:
  nuevo2[column].replace({"Discusión única": 4,"Discusión informe de Comisión Mixta":5}, inplace=True)

for column in nuevo2:
  nuevo2[column].replace({"Mayoría simple": 1,"Cuatro séptimos Q.C.":2,"Q.C.":3,"Tres quintos Q.C.":4,"Dos tercios Q.C.":5}, inplace=True)

nuevo2 = nuevo2.filter(items=['Latorre R., Juan Ignacio',  'ETAPA', 'QUORUM','TIPOVOTACION'])





######################

# Obtenemos variables independientes
X_ = nuevo2.drop(["Latorre R., Juan Ignacio"],axis = 1)



# Obtenemos variable dependiente
Y_ = nuevo2["Latorre R., Juan Ignacio"]




# Obtenemos variables independientes
X_ = nuevo2.drop(["Latorre R., Juan Ignacio"],axis = 1)


# Obtenemos variable dependiente
Y_ = nuevo2["Latorre R., Juan Ignacio"]


X_train, X_test, y_train, y_test = train_test_split(X_, Y_, test_size=0.2, stratify=Y_)

clf = RandomForestClassifier()
clf.fit(X_train, y_train).score(X_test, y_test)


#checkpoints
#C:\Users\Pc\Downloads\Iris_Heroku-master\checkpoints\hola.pkl
filename = "asistencia.pkl"
pickle.dump(clf, open(filename, "wb"))


loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X_test, y_test)
print(result)
print(loaded_model.predict([[2,1,3]]))


