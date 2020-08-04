import pandas as pd
import ast
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


class Machine:
    def __init__(self):
        self._data = pd.read_csv(r'Data/VENDAS_NF_TRAIN.csv')
        self._x, self._y = self.__modeling_data(self._data)
        self.train_data(self._x, self._y)
        #self.classifier
        print('MODELO TREINADO!')


    def __transform_input(self, data):
        data = ast.literal_eval(data)
        df = pd.DataFrame(data=data)
        x = self.__modeling_predictors(df)

        return x


    def __modeling_predictors(self, data):
        x = data.iloc[:, 0:7].values

        sc = StandardScaler()
        x[:, 3:6] = sc.fit_transform(x[:, 3:6])

        label = LabelEncoder()
        x[:, 1] = label.fit_transform(x[:, 1])

        return x

    
    def __modeling_data(self, data):
        x = self.__modeling_predictors(data)
        y = data.iloc[:, 7].values

        return (x, y)


    def show_data(self):
        return print(self.data.head())
    

    def train_data(self, x, y):
        global classifier
        classifier = RandomForestClassifier(n_estimators=200, random_state=0,)
        classifier.fit(x, y)
    

    def predict(self, data):
        y = self.__transform_input(data)

        return classifier.predict(y)
