import numpy as np
from sklearn import linear_model
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

csv = np.genfromtxt ('train.csv', delimiter=",")

x = csv[1:501,2:12]
y = csv[1:501,1]


kf = KFold(n_splits=10)
lam = [0.1, 1 ,10,100,1000]
for j in range(0 , 5):
    i = 0
    rmse = np.zeros(10)
    for train_index, test_index in kf.split(x):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model = linear_model.Ridge(alpha=lam[j])
        model.fit(x_train,y_train)
        estimate = model.predict(x_test)
        rmse[i] = np.sqrt(mean_squared_error(estimate,y_test))
        i=i+1
    #endfor

    avg_rmse = np.mean(rmse)
    print(lam[j], avg_rmse)
#endfor




