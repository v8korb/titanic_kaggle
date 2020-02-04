# Titanic Kaggle

O desafio constitui-se em propor um modelo preditivo que seja capaz de avaliar a probabilidade dos tripulantes de acordo com os dados propostos.
São disponibilizados dois datasets, um de treinamento e outro de testes (.csv) 
```
train: 891 passageiros. Revelado se sobreviveram ou não.
test: 418 passageiros. Não revelado se sobreviveram.
features: 10 (4 string, 2 float, 3 integers, 1 binary)
```

## Pré-processamento
- Avaliação das features: separação em categóricas e não categóricas, tipos e descrições
- Remoção de features consideradas de pouco valor​ (Name,Ticket,Fare,Cabin,PassengerId)
- Preenchimento de valores nulos: preenchimento de valores nulos da coluna idade com a mediana
- Preenchimento de valores não nulos com a moda na feature EMBARKED
- Histogramas de relação entre a sobrevivencia e as features.
 
## Modelos k-nearest neighbors algorithm (knn) 
Pode ser usado para classificar os passageiros e prever valores de sobrevivência. Foram testados os modelos kNN com k=1, k=3, k=5, distância Euclidiana e distância de Manhatan. Foi usada validação cruzada com 3 folds.


## Modelo de rede neural
A camada de saída possui uma função de ativação sigmóide para que a saída seja 0 ou 1. Temos uma camada de entrada com 10 nós que alimenta uma camada oculta com 8 nós e uma camada de saída que é usada para prever a sobrevivência de um passageiro.

## Requisitos
- Pandas
- Seaborn
- Sklearn
- WordCloud
- Tensorflow
- Keras
