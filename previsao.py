"""Original file is located at
    https://colab.research.google.com/drive/1pug-TrAy3rI7EkE4zWtbT7CayXH-Yxhw
"""

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import seaborn as sns

dados = pd.read_csv('Wellbeing_and_lifestyle_data_Kaggle.csv')

#dados.corr().round(4)
#Melhores Correlações para ACHIEVEMENT: WORK_LIFE_BALANCE_SCORE(0.5612) / PERSONAL_AWARDS(0.3962) / FLOW(0.3866) / TIME_FOR_PASSION(0.3689) / SUPPORTING_OTHERS(0.3601) / LIVE_VISION(0.3207)
#PERSONAL_AWARDS: WORK_LIFE_BALANCE_SCORE(0.5042) / ACHIEVEMENT(0.3962) / SUPPORTING_OTHERS (0.3327) / DONATION(0.2778)
#WORK_LIFE_BALANCE_SCORE: ACHIEVEMENT(0.5612) / TIME_FOR_PASSION(0.5170) / PERSONAL_AWARDS(0.5042) / SUPPORTING_OTHERS(0.5489) / TODO_COMPLETED(0.5455) / PLACES_VISITED(0.5296) / FRUITS_VEGGIES(0.4523)

import pandas as pd
from sklearn.linear_model import LinearRegression

# import do dataset
dados = pd.read_csv('Wellbeing_and_lifestyle_data_Kaggle.csv')

# Dados de entrada
df = pd.DataFrame(dados)

# Separação dos dados em X (atributos) e y (variável alvo)
X = df[['ACHIEVEMENT', 'SUPPORTING_OTHERS', 'FLOW', 'CORE_CIRCLE', 'PLACES_VISITED', 'TODO_COMPLETED', 'TIME_FOR_PASSION', 'FRUITS_VEGGIES', 'PERSONAL_AWARDS']]
y = df['WORK_LIFE_BALANCE_SCORE']

# Treinamento do modelo de regressão linear
model = LinearRegression()
model.fit(X, y)

# Entrada dos valores para o novo dado da previsão
new_data = {}

for attribute in X.columns:
    value = float(input())
    new_data[attribute] = [value]

new_df = pd.DataFrame(new_data)

# Realizando a previsão
prediction = model.predict(new_df)

# Exibindo a previsão
print(prediction[0])

#mse = mean_squared_error(y, model.predict(X))
#rmse = mean_squared_error(y, model.predict(X), squared=False)
#r2 = r2_score(y, model.predict(X))

#print("MSE:", mse)
#print("RMSE:", rmse)
#print("R2:", r2)

