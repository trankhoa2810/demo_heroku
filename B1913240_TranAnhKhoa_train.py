# Mo hinh phan lop hoa iris bang thuat toan cay quyet dinh (Decision Tree).

import pandas as pd
import numpy as np
from pandas.core.frame import DataFrame

def training_Module(sepalLength, sepalWidth, petalLength, petalWidth):
    data = pd.read_csv("iris.csv")
    X = data.iloc[:,:-1] # gan gia tri 4 cot dau tien cua dataset cho bien X.
    y = data['variety'] # gan gia tri cot du doan cho bien y.

    from sklearn.model_selection import train_test_split
    # Chia tap du lieu thanh 2 phan:
    # 90% train va 10% test.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    from sklearn.tree import DecisionTreeClassifier
    model = DecisionTreeClassifier(criterion='gini')
    model.fit(X_train, y_train)

    result = model.predict([[sepalLength, sepalWidth, petalLength, petalWidth]])
    return str(np.asarray(result))


# y_preds = model.predict(X_test.head())

# So sanh ket qua du doan du lieu sau khi train
# print(pd.DataFrame({"y_real ": y_test.head(), "y_predict" : y_preds}))

# Danh gia do chinh xac mo hinh.
# from sklearn.metrics import accuracy_score
# print("Accuracy Score: {}".format(accuracy_score(y_test.head(), y_preds,)))