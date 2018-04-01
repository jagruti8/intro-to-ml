import numpy as np
from sklearn import linear_model
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


csv = np.genfromtxt ('train.csv', delimiter=",")

x_d = csv[1:901,2:7]
y = csv[1:901,1]

x_sq = np.square(x_d)
x_eq = np.exp(x_d)
x_cos = np.cos(x_d)
x_1 = np.ones((900,1))*1.0
x = np.concatenate((x_d,x_sq,x_eq,x_cos,x_1),1)
#x_train, x_test, y_train, y_test = train_test_split(
#    x, y, test_size=0.22, random_state=42)

#print(x[0,:])

'''
kf = KFold(n_splits=10)
i = 0
rmse = np.zeros(10)
rmse1 = np.zeros(10)
rmse2 = np.zeros(10)
coef = np.zeros((10,21))
intercept = np.zeros(10)
for train_index, test_index in kf.split(x_train):
    x_train_1, x_test_1 = x_train[train_index], x_train[test_index]
    y_train_1, y_test_1 = y_train[train_index], y_train[test_index]
    # model = linear_model.LinearRegression(fit_intercept=False)
    #model= linear_model.Lasso(alpha=0.1, fit_intercept=False,
     #                 precompute=False, copy_X=True, max_iter=1000, tol=0.0001,
      #                warm_start=False, positive=False, random_state=None, selection='cyclic')
    model = linear_model.LassoCV(eps=0.001, n_alphas=100, alphas=None, fit_intercept=True, normalize=False, precompute='auto',
                    max_iter=1000, tol=0.0001, copy_X=True, cv=None,
                    verbose=False, n_jobs=1, positive=False, random_state=None, selection='cyclic')
    model.fit(x_train_1,y_train_1)
    coef[i,:]=model.coef_
    intercept[i]=model.intercept_
    estimate = model.predict(x_test_1)
    estimate1 = model.predict(x_test)
    estimate2 = model.predict(x)
    rmse[i] = np.sqrt(mean_squared_error(estimate,y_test_1))
    rmse1[i] = np.sqrt(mean_squared_error(estimate1,y_test))
    rmse2[i] = np.sqrt(mean_squared_error(estimate2,y))
    print(rmse[i])
    print(rmse1[i])
    print(rmse2[i])
    #print(coef[i,:])
    #print(model.alpha_)
    i=i+1
print(np.argmin(rmse))
print(np.argmin(rmse1))
print(np.argmin(rmse2))
coef_final_index = np.argmin(rmse1)
coef_final = coef[coef_final_index,:]
print(coef_final)

#np.savetxt("sample_lasso_cv_1.csv", coef_final, delimiter=",")

#avg_coef = np.zeros((1,21))
#print(avg_coef.shape)
#avg_coef = np.mean(coef,0)
#print(avg_coef.shape)
#avg_intercept = np.mean(intercept)
#print(avg_coef)
#print(avg_intercept)
#avg_rmse = np.mean(rmse)
#y_1_predict = np.zeros((198,21))
#print(x_test.shape)
#y_1_predict = np.multiply(x_test,avg_coef,0)



'''

kf = KFold(n_splits=10)
#lam = np.random.uniform(0.1,0.2,200)
lam = [0.1,0.131165552252737, 0.2,1 ,10,100,1000]
#print(lam)
#exit()
avg_rmse = np.zeros((7,1))
avg_var = np.zeros((7,1))
for j in range(0 , 7):
    i = 0
    rmse = np.zeros(7)
    '''
    for train_index, test_index in kf.split(x):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model = linear_model.Lasso(alpha=lam[j])
        model.fit(x_train,y_train)
        estimate = model.predict(x_test)
        rmse[i] = np.sqrt(mean_squared_error(estimate,y_test))
        i=i+1


    avg_rmse[j] = np.mean(rmse)
    avg_var[j] = np.var(rmse)
    print(lam[j], avg_rmse[j], avg_var[j])

    '''
    model = linear_model.Lasso(alpha=lam[j], fit_intercept=True,
                            precompute=True, copy_X=True, max_iter=1000, tol=0.0001,
                           warm_start=False, positive=False, random_state=None, selection='cyclic')
    model.fit(x, y)
    estimate = model.predict(x)
    rmse[j] = np.sqrt(mean_squared_error(estimate, y))
#model1 = linear_model.LinearRegression(fit_intercept=False)
rmse1 = np.argsort(rmse)
#var = np.argmin(avg_var)
print(rmse1)
#print(lam[var])
model1 = linear_model.Lasso(alpha=0.1, fit_intercept=True,
                            precompute=True, copy_X=True, max_iter=1000, tol=0.0001,
                           warm_start=False, positive=False, random_state=None, selection='cyclic')
model2 = linear_model.LassoCV(eps=0.001, n_alphas=100, alphas=None, fit_intercept=True, normalize=False,
                             precompute='auto',
                             max_iter=1000, tol=0.0001, copy_X=True, cv=10,
                             verbose=False, n_jobs=1, positive=False, random_state=None, selection='cyclic')

a=model1.fit(x,y)
b=model2.fit(x,y)
print(model1.coef_)
print(model2.coef_)
#print(model1.intercept_)
print(model2.alpha_)

#np.savetxt("sample_lasso_no_normalize.csv", model1.coef_, delimiter=",")


y_pred1 = model1.predict(x)
y_pred2 = model2.predict(x)
print(np.sqrt(mean_squared_error(y, y_pred1)))
print(np.sqrt(mean_squared_error(y, y_pred2)))


#x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.98, random_state=None)
#model2 = linear_model.LinearRegression()
#model2.fit(x_train,y_train)
#print(model2.coef_)

