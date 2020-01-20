import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor

DataDictionary = {'survival': '0/1' ,'PassengerId': 'NumberId' ,'pclass': 'Ticket class(1,2,3)' ,
 'Name': 'name' ,'sex': 'male/female', 'Age': 'age' , 'SibSp': '#siblings/spouses aboard' , 
 'Parch': '#ofparents/children aboard', 'Ticket': 'ticket number', 'Fare': '$', 
 'Cabin': 'CabinNumber' ,'Embarked': 'Port of Embarkation' }


data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")
X = data.copy()
test_X = test_data.copy()
#X = X.drop(['Survived'], axis = 1)
all_features = ['PassengerId','pclass', 'Name', 'sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin' ,'Embarked']
#y = data['Survived']


#FILTRADO Y LIMPIADO DE DATOS 
    #Missing Values 
#Veo que columnas tienen columnas vacias 
cols_with_missing = [col for col in X.columns if X[col].isnull().any() ]
cols_with_missing_test = [col for col in test_X.columns if test_X[col].isnull().any() ] 
#print(cols_with_missing_test) #AGE , FARE Y CABIN
#print(cols_with_missing) #AGE, CABIN Y EMBARKED


features1 = ['PassengerId','pclass', 'Name' ,'sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare','Embarked']

#Para simplificar, voy a eliminar la columna Cabin, la columna Name (aporta algo?), la columna Ticket(aporta algo?), las filas vacias de Age y Embarked y las filas con fare = 0

#EMBARKED
X_Dropped = X.dropna(subset = ['Embarked'], axis = 0) #elimino unicamente filas con NaN , son 2 solamente 
check0 = [row for row in X_Dropped['Embarked'] if X_Dropped['Embarked'].isnull().any() ] #chequeo si quedaron filas con espacios vacios 


#CABIN , NAME , TICKET
#elimino toda la columna
cols_to_delete = ['Cabin', 'Name' , 'Ticket']
X_Dropped = X_Dropped.drop(cols_to_delete, axis = 1)
test_X_Dropped = test_X.drop(cols_to_delete , axis = 1)


#FARE
X_Dropped.drop(X_Dropped.loc[X_Dropped['Fare'] == 0].index, inplace=True) #chequear bien esta linea !!!
test_X_Dropped.drop(test_X_Dropped.loc[test_X_Dropped['Fare'] == 0].index, inplace=True)


features2 = ['PassengerId' ,'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare','Embarked']

#Separating numeric and categorical features
categ_feat = [col for col in features2 if X_Dropped[col].dtype == 'object']
num_feat = list(set(features2) - set(categ_feat))


#Para hacer imputation tengo que pasar primero las columnas con Sex y Embarked a valores numericos - LabelEncoding
#SEX & EMBARKED -LABEL ENCODING
le = LabelEncoder()
label_X = X_Dropped.copy()
label_X_test = test_X_Dropped.copy()

for col in categ_feat:
    label_X[col] = le.fit_transform(X_Dropped[col])
    label_X_test[col] = le.transform(test_X_Dropped[col])

#AGE - Imputation 
X_plus = label_X.copy()
test_X_plus = label_X_test.copy()
#Make new columns indicating what will be imputed
X_plus['Age' + '_was_missing'] = X_plus['Age'].isnull()
for col in cols_with_missing_test[:1]:
    test_X_plus [col + '_was_missing'] = test_X_plus[col].isnull()
#Impute
imputer = SimpleImputer()
imputed_X = pd.DataFrame(imputer.fit_transform(X_plus))
imputed_X_test = pd.DataFrame(imputer.fit_transform(test_X_plus))

imputed_X.columns = X_plus.columns
imputed_X_test.columns = test_X_plus.columns
final_X_test = imputed_X_test

final_y = imputed_X['Survived']
final_X = imputed_X.drop(['Survived'] , axis = 1)

#Division de data para VALIDACION
train_X , val_X , train_y , val_y = train_test_split(final_X , final_y , train_size = 0.8 , test_size = 0.2, random_state = 0)

#SHAPES
print('Test X Size: ', imputed_X_test.shape  )
print('Val X Size: ' , val_X.shape , '    ' , 'Val y Size: ' , val_y.shape )
print('Train X Size: ' , train_X.shape , '     ' , 'Train y Size: ' , train_y.shape )

#Modelo 
model = RandomForestRegressor()
model.fit(train_X , train_y)
val_pred = model. predict(val_X)
print(val_y , val_pred)
test_pred = model.predict(final_X_test)

#print(test_pred)
#Validating
my_mae = mean_absolute_error(val_y , val_pred)

#print(my_mae)


#DUDAS
#Porque no fitteo los datos a validad? solo les hago transform()


#Tips
#bad_rows = X.loc[X['Fare'] == 0.0000] #localiza las filas donde coincida la condicion / 15
#print(bad_rows.iloc[0,0]) - Buena manera para sacar un dato especifico si conozco la posicion o para recorrer y buscarlo tambien 
#print(X_Dropped.loc[X_train_Dropped['Age'].isnull()]) #64 - mejor opcion me parece - Imputation
#train_X , val_X , train_y , val_y = train_test_split(imputed_X , y) #---> aca esta el error, si hago la division para la validacion despues de limpiar la data, como se que filas de y remover para que quede con el mismo tamano!!!
