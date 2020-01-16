import pandas as pd

DataDictionary = {'survival': '0/1' ,'pclass': 'Ticket class(1,2,3)' ,
 'sex': 'male/female', 'Age': 'age' , 'SibSp': '#siblings/spouses aboard' , 
 'Parch': '#ofparents/children aboard', 'Ticket': 'ticket number', 'Fare': '$', 
 'Cabin': 'CabinNumber' ,'Embarked': 'Port of Embarkation' }


data = pd.read_csv("train.csv")
X_train = data.copy()
X_train = X_train.drop(['Survived'], axis = 1)
y = data['Survived']
#print(X_train)
#print(y)


#Missing Values 
#Veo que columnas tienen columnas vacias 
cols_with_missing = [col for col in X_train.columns if X_train[col].isnull().any() ]
#print(X_train[cols_with_missing])

#print(X_train.describe())

xx = X_train.loc[X_train['Fare'] == 0]
xy = [row for row in X_train['Embarked'] if X_train['Embarked'].isnull().any()] #chequeo si tiene vacias 
#print(xy)

#Para simplificar ahora nomas, voy a eliminar la columna Cabin, las filas vacias de Age y Embarked y las filas con fare = 0
#EMBARKED
X_train_Dropped = X_train.dropna(subset = ['Embarked'], axis = 0) #elimino unicamente filas con NaN , son 2 solamente 
check0 = [row for row in X_train_Dropped['Embarked'] if X_train_Dropped['Embarked'].isnull().any() ] #chequeo si quedaron filas con espacios vacios 
#print(check0)

#CABIN
X_train_Dropped = X_train_Dropped.drop(['Cabin'] , axis = 1) #elimino toda la columna

#FARE
bad_rows = X_train.loc[X_train['Fare'] == 0.0000] #localiza las filas donde coincida la condicion / 15
bad_rows = dict(bad_rows)
#X_train_Dropped = X_train_Dropped.set_index('Fare')
#X_train_Dropped = X_train_Dropped.drop(bad_rows['PassengerId'], axis = 0)
#print(bad_rows['PassengerId'])
print(bad_rows)

#AGE

#print(X_train_Dropped.loc[X_train_Dropped['Age'] == 0])


