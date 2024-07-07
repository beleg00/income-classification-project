# Income Classification Project

This project demonstrates a basic machine learning pipeline to classify income levels based on census data. The code is written in Python and leverages libraries such as Pandas, NumPy, Matplotlib, Seaborn, and Scikit-learn.

## Table of Contents

1. [Installation](#installation)
2. [Data Loading](#data-loading)
3. [Data Exploration](#data-exploration)
4. [Data Preprocessing](#data-preprocessing)
5. [Model Training](#model-training)
6. [Evaluation](#evaluation)
7. [Visualization](#visualization)
8. [Contributing](#contributing)
9. [License](#license)

## Installation

To run this project, you need to have Python installed. You can install the required packages using pip:

```sh
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Data Loading

The dataset used is the Adult dataset from the UCI Machine Learning Repository. The data is loaded into a Pandas DataFrame from a URL.

```python
import pandas as pd

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
column_names = [
    'age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
    'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
    'hours-per-week', 'native-country', 'income'
]
data = pd.read_csv(url, names=column_names, na_values=' ?', skipinitialspace=True)
```

## Data Exploration

Exploratory Data Analysis (EDA) is performed to understand the dataset better. This includes displaying the first few rows, summary statistics, and value counts of the target variable.

```python
data.head()
data.info()
data.describe()
data['income'].value_counts()
```

## Data Preprocessing

Data preprocessing includes handling missing values, encoding categorical features, and scaling numerical features.

```python
# Checking for missing values
data.isnull().sum()

# Dropping rows with missing values
data.dropna(inplace=True)

# Alternatively, fill missing values with the most frequent value
# data.fillna(data.mode().iloc[0], inplace=True)

# Splitting features and target variable
X = data.drop('income', axis=1)
y = data['income']

# Converting the target variable to binary
y = y.apply(lambda x: 1 if x == '>50K' else 0)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

categorical_features = X.select_dtypes(include=['object']).columns
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])
```

## Model Training

Two machine learning models are trained: Logistic Regression and Support Vector Machine (SVM).

```python
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# Logistic Regression Pipeline
logreg_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                  ('classifier', LogisticRegression(max_iter=1000))])

# Support Vector Machine Pipeline
svc_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', SVC(kernel='linear'))])

# Training Logistic Regression
logreg_pipeline.fit(X_train, y_train)

# Training Support Vector Machine
svc_pipeline.fit(X_train, y_train)
```

## Evaluation

The accuracy of both models is evaluated on the test set.

```python
from sklearn.metrics import accuracy_score

logreg_predictions = logreg_pipeline.predict(X_test)
logreg_accuracy = accuracy_score(y_test, logreg_predictions)
print(f'Logistic Regression Accuracy: {logreg_accuracy:.4f}')

svc_predictions = svc_pipeline.predict(X_test)
svc_accuracy = accuracy_score(y_test, svc_predictions)
print(f'Support Vector Machine Accuracy: {svc_accuracy:.4f}')
```

## Visualization

Visualizations are created to understand the distribution of income by age and sex.

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Distribution of income by age
plt.figure(figsize=(10, 6))
sns.histplot(data=data, x='age', hue='income', multiple='stack')
plt.title('Distribution of Income by Age')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()

# Distribution of income by sex
plt.figure(figsize=(10, 6))
sns.countplot(data=data, x='sex', hue='income')
plt.title('Distribution of Income by Sex')
plt.xlabel('Sex')
plt.ylabel('Count')
plt.show()
```

## Contributing

Contributions are welcome! Please submit a pull request or open an issue to discuss any changes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

This README provides an overview of the project, instructions for installation, and details on how to run and evaluate the code. Feel free to modify it to suit your specific needs.
