import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

DataDictionary = {'survival': '0/1' ,'PassengerId': 'NumberId' ,'pclass': 'Ticket class(1,2,3)' ,
 'Name': 'name' ,'sex': 'male/female', 'Age': 'age' , 'SibSp': '#siblings/spouses aboard' , 
 'Parch': '#ofparents/children aboard', 'Ticket': 'ticket number', 'Fare': '$', 
 'Cabin': 'CabinNumber' ,'Embarked': 'Port of Embarkation' }


data = pd.read_csv("train.csv")
X = data.copy()
X = X.drop(['Survived'], axis = 1)
all_features = ['PassengerId','pclass', 'Name', 'sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin' ,'Embarked']
y = data['Survived']

#Division de data para VALIDACION
train_X , val_X , train_y , val_y = train_test_split(X , y , train_size = 0.8 , test_size = 0.2, random_state = 0)

#FILTRADO Y LIMPIADO DE DATOS 
    #Missing Values 
#Veo que columnas tienen columnas vacias 
cols_with_missing_train = [col for col in train_X.columns if train_X[col].isnull().any() ]
cols_with_missing_valid = [col for col in val_X.columns if val_X[col].isnull().any() ]
#print(cols_with_missing_train) #tiene las 3 columnas faltantes
#print(cols_with_missing_valid) #tiene solo dos, embarked esta completa

features1 = ['PassengerId','pclass', 'Name' ,'sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare','Embarked']

#Para simplificar, voy a eliminar la columna Cabin, la columna Name (aporta algo?), la columna Ticket(aporta algo?), las filas vacias de Age y Embarked y las filas con fare = 0

#EMBARKED
train_X_Dropped = train_X.dropna(subset = ['Embarked'], axis = 0) #elimino unicamente filas con NaN , son 2 solamente 
check0 = [row for row in train_X_Dropped['Embarked'] if train_X_Dropped['Embarked'].isnull().any() ] #chequeo si quedaron filas con espacios vacios 


#CABIN , NAME , TICKET
#elimino toda la columna
cols_to_delete = ['Cabin', 'Name' , 'Ticket']
train_X_Dropped = train_X_Dropped.drop(cols_to_delete, axis = 1)
val_X_Dropped = val_X.drop(cols_to_delete, axis = 1)


#FARE
train_X_Dropped.drop(train_X_Dropped.loc[train_X_Dropped['Fare'] == 0].index, inplace=True) #chequear bien esta linea !!!
val_X_Dropped.drop(val_X_Dropped.loc[val_X_Dropped['Fare'] == 0].index, inplace=True)

features2 = ['PassengerId' ,'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare','Embarked']

#Separating numeric and categorical features
categ_feat = [col for col in features2 if train_X_Dropped[col].dtype == 'object']
num_feat = list(set(features2) - set(categ_feat))


#Para hacer imputation tengo que pasar primero las columnas con Sex y Embarked a valores numericos - LabelEncoding
#SEX & EMBARKED -LABEL ENCODING
le = LabelEncoder()
label_X_train = train_X_Dropped.copy()
label_X_val = val_X_Dropped.copy()
for col in categ_feat:
    label_X_train[col] = le.fit_transform(train_X_Dropped[col])
    label_X_val[col] = le.transform(val_X_Dropped[col])
#AGE - Imputation 
train_X_plus = label_X_train.copy()
val_X_plus = label_X_val.copy()
#Make new columns indicating what will be imputed
train_X_plus['Age' + '_was_missing'] = train_X_plus['Age'].isnull()
val_X_plus['Age' + '_was_missing'] = val_X_plus['Age'].isnull()
#Impute
imputer = SimpleImputer()
imputed_X_train = pd.DataFrame(imputer.fit_transform(train_X_plus))
imputed_X_val = pd.DataFrame(imputer.transform(val_X_plus))

imputed_X_train.columns = train_X_plus.columns
imputed_X_val.columns = val_X_plus.columns





#DUDAS
#Porque no fitteo los datos a validad? solo les hago transform()


#Tips
#bad_rows = X.loc[X['Fare'] == 0.0000] #localiza las filas donde coincida la condicion / 15
#print(bad_rows.iloc[0,0]) - Buena manera para sacar un dato especifico si conozco la posicion o para recorrer y buscarlo tambien 
#print(X_Dropped.loc[X_train_Dropped['Age'].isnull()]) #64 - mejor opcion me parece - Imputation
#train_X , val_X , train_y , val_y = train_test_split(imputed_X , y) #---> aca esta el error, si hago la division para la validacion despues de limpiar la data, como se que filas de y remover para que quede con el mismo tamano!!!
