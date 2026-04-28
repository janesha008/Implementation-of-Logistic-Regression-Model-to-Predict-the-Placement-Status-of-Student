# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. Load the dataset, remove unnecessary columns, and convert categorical data into numerical format.
2. Split the dataset into training and testing sets.
3. Train a Logistic Regression model using the training data and evaluate its accuracy.
4. Plot the relationship between a selected feature and predicted probabilities using a logistic curve.

## Program:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("Placement_Data.csv")
df = df.drop("salary", axis=1)
df = pd.get_dummies(df, drop_first=True)

x = df.drop("status_Placed", axis=1)
y = df['status_Placed']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)
print(f"Accuracy: {model.score(x_test, y_test)}")

X = x.iloc[:, 0].values.reshape(-1, 1)
model_plot = LogisticRegression(max_iter=1000)
model_plot.fit(X, y)

plt.scatter(X, y, color="cyan", label="Data Points")
x_val = np.linspace(X.min(), X.max(), 100)
y_val = model_plot.predict_proba(x_val.reshape(-1, 1))[:, -1]
plt.plot(x_val, y_val, color="red", label="Fit Line")

plt.xlabel("Feature")
plt.ylabel("Probability")
plt.title("Logistic Regression")
plt.legend()
plt.show()
```

## Output:

<img width="732" height="597" alt="image" src="https://github.com/user-attachments/assets/75da9436-6ca2-4991-846a-21008d9fe36f" />


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
