# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 22:46:25 2024

@author: beleg00
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Загрузка данных
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
column_names = [
    'age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
    'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
    'hours-per-week', 'native-country', 'income'
]
data = pd.read_csv(url, names=column_names, na_values=' ?', skipinitialspace=True)

data.head()

data.info()
data.describe()
data['income'].value_counts()

# Проверка на пропуски
data.isnull().sum()

# Удаление строк с пропусками
data.dropna(inplace=True)

# Альтернативный способ: заполнение пропусков наиболее частыми значениями
# data.fillna(data.mode().iloc[0], inplace=True)

# Распределение дохода по возрасту
plt.figure(figsize=(10, 6))
sns.histplot(data=data, x='age', hue='income', multiple='stack')
plt.title('Distribution of income by age')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()

# Распределение дохода по полу
plt.figure(figsize=(10, 6))
sns.countplot(data=data, x='sex', hue='income')
plt.title('Distribution of income by sex')
plt.xlabel('Sex')
plt.ylabel('Count')
plt.show()

# Выделение целевой переменной и признаков
X = data.drop('income', axis=1)
y = data['income']

# Преобразование целевой переменной в бинарную форму
y = y.apply(lambda x: 1 if x == '>50K' else 0)

# Разделение данных на тренировочную и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Преобразование категориальных признаков с помощью OneHotEncoder
categorical_features = X.select_dtypes(include=['object']).columns
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

# Создание пайплайна для логистической регрессии
logreg_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                  ('classifier', LogisticRegression(max_iter=1000))])

# Создание пайплайна для метода опорных векторов
svc_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', SVC(kernel='linear'))])


# Обучение логистической регрессии
logreg_pipeline.fit(X_train, y_train)
logreg_predictions = logreg_pipeline.predict(X_test)
logreg_accuracy = accuracy_score(y_test, logreg_predictions)
print(f'Точность логистической регрессии: {logreg_accuracy:.4f}')

# Обучение метода опорных векторов
svc_pipeline.fit(X_train, y_train)
svc_predictions = svc_pipeline.predict(X_test)
svc_accuracy = accuracy_score(y_test, svc_predictions)
print(f'Точность метода опорных векторов: {svc_accuracy:.4f}')
