# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner

## Algorithm
1.Import the standard libraries.
2.Upload the dataset and check for any null or duplicated values using .isnull() and .duplicated() function respectively.
3.Import LabelEncoder and encode the dataset.
4.Import LogisticRegression from sklearn and apply the model on the dataset.
5.Predict the values of array.
6.Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.
7.Apply new unknown values

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Easwar J
RegisterNumber: 212221230024
*/
```
```
import pandas as pd
data = pd.read_csv("Placement_Data.csv")
data.head()
data1 = data.copy()
data1 = data1.drop(["sl_no","salary"],axis = 1)
data1.head()
data1.isnull().sum()
data1.duplicated().sum()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data1["gender"] = le.fit_transform(data1["gender"])
data1["ssc_b"] = le.fit_transform(data1["ssc_b"])
data1["hsc_b"] = le.fit_transform(data1["hsc_b"])
data1["hsc_s"] = le.fit_transform(data1["hsc_s"])
data1["degree_t"] = le.fit_transform(data1["degree_t"])
data1["workex"] = le.fit_transform(data1["workex"])
data1["specialisation"] = le.fit_transform(data1["specialisation"])
data1["status"] = le.fit_transform(data1["status"])
data1
x = data1.iloc[:,:-1]
x
y = data1["status"]
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver = "liblinear")
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
y_pred
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
accuracy
from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
confusion
from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
classification_report1
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```

## Output:
### HEAD:
![ml1](https://user-images.githubusercontent.com/94154683/163658991-0fafae2c-27d2-4988-814b-ddd0aa9b4f4b.png)


### PREDICTED VALUES:
![ml2](https://user-images.githubusercontent.com/94154683/163658993-2a23d3d8-713a-4f35-a25d-9f8442ff7c65.png)


### ACCURACY:
![ml3](https://user-images.githubusercontent.com/94154683/163658998-4aa4a4f5-c4ba-4ecd-b31d-e877e5c600cf.png)


### CONFUSION MATRIX:
![ml4](https://user-images.githubusercontent.com/94154683/163659002-81ee2d87-1d1f-4ad4-9a30-261f7a28442f.png)


### CLASSIFICATION REPORT:
![ml5](https://user-images.githubusercontent.com/94154683/163659006-922155c9-46f4-4072-ada7-5728c383bc44.png)



## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
