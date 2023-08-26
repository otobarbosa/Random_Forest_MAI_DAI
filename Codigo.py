# Biblioteca para a modelagem de dados
import pandas as pd

# Bilioteca para recursos matemáticos
import numpy as np

# Bilbioteca de plotagem de dados
import seaborn as sns
import matplotlib.pyplot as plt

# Biblioteca/Função para ignorar avisos
# from warnings import filterwarnings

# Plan1 = Base de Dados
# Plan2 = Novas Entradas
Base_Dados = pd.read_excel('caminho da planilha', 'Plan1')

Base_Dados.head()
Base_Dados.info()
Base_Dados.describe()

sns.set( font_scale=1.5, rc={'figure.figsize':(20,20)})
eixo = Base_Dados.hist(bins=20, color = 'purple')

fig = plt.figure( figsize = (10,5))
fig.suptitle("RECALL")
sns.boxplot( data = Base_Dados);

# Comando iloc[linhas, colunas]

Caracteristicas = Base_Dados.iloc[:, 1:4].values
Previsor = Base_Dados.iloc[:, 4:5].values

from sklearn.model_selection import train_test_split
x_treinamento, x_teste, y_treinamento, y_teste = train_test_split(
    Caracteristicas,
    Previsor,
    test_size=0.30,
    random_state=10
)
print(len(Base_Dados))
print(len(x_treinamento))
print(len(x_teste))

from sklearn.ensemble import RandomForestRegressor
# Ou trocar RandomForestRegressor por RandomForestClassier, para classificação
Algoritmo_floresta_aleatoria = RandomForestRegressor( n_estimators = 500)
Algoritmo_floresta_aleatoria.fit( x_treinamento, y_treinamento )

Previsoes = Algoritmo_floresta_aleatoria.predict( x_teste)

from sklearn.metrics import confusion_matrix
Matriz_confusao = confusion_matrix(y_teste, Previsoes)
print(Matriz_confusao)

plt.figure( figsize=(10,5))
sns.heatmap( Matriz_confusao, annot = True)

from sklearn.metrics import classification_report
report = classification_report(y_teste, Previsoes)
print(report)

predicao_atual = pd.read_excel('caminho da planilha', 'Plan2')
predicao_atual.head()

Prever = predicao_atual.iloc[:, 1:4].values

predicao_atual['Previsao do Modelo'] = Algoritmo_floresta_aleatoria.predict(Prever)

predicao_atual['Previsao do Modelo'].value_counts()

predicao_atual
