import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn import svm
from numpy import unravel_index

# Read data
df_train = pd.read_csv ('train.csv', delimiter=",",header=0)
df_test = pd.read_csv ('test.csv', delimiter=",",header=0)
#df_train.info()
#df_test.info()

train_y = df_train['y']
train_x = df_train.drop(['Id', 'y'], axis=1, inplace=False)
test_x= df_test.drop(['Id'], axis=1, inplace=False)
#df_train.info()
#train_x.info()
#df_test.info()
#test_x.info()

# Feature selection using Random Forest
clf = RandomForestClassifier(bootstrap=True,max_depth=None, random_state=0,class_weight='balanced',max_features='auto',
                             criterion='entropy',min_samples_split=5,min_samples_leaf=1)
clf.fit(train_x, train_y)
imp_feat_rf = pd.Series(clf.feature_importances_, index=train_x.columns).sort_values(ascending=False)
indices = np.argsort(clf.feature_importances_)[::-1]

# List the names of the names of top n selected features and remove the unicode
n = 10
select_feat =[str(s) for s in train_x.columns[indices][:n]]
print(select_feat)

# Make the subsets with n features only
train_x_sub = train_x[select_feat].values
test_sub = test_x[select_feat].values
print(train_x_sub.shape)
print(test_sub.shape)

# Plot feature importance
imp_feat_rf[:n].plot(kind='bar', title='Feature Importance with Random Forest', figsize=(12,8))
plt.ylabel('Feature Importance values')
plt.subplots_adjust(bottom=0.25)
plt.savefig('FeatImportance.png')
plt.show()

# Cross validation for choosing SVM parameters
skf = StratifiedKFold(n_splits=10)
skf.get_n_splits(train_x_sub, train_y)
avg_accuracy = np.zeros(((2,4,3)))
print(avg_accuracy.shape)
C = [1, 10, 100, 1000]
gamma = [0.0001, 0.001, 0.01]
kernel=['rbf','sigmoid']
for j in range(0,1):
    for k in range(0,4):
        for l in range(0,3):
            i=0
            accuracy=np.zeros(10)
            for train_index, test_index in skf.split(train_x_sub,train_y):
                x_train_1, x_test_1 = train_x_sub[train_index], train_x_sub[test_index]
                y_train_1, y_test_1 = train_y[train_index], train_y[test_index]
                svc = svm.SVC(C=C[k], kernel=kernel[j], gamma=gamma[l], class_weight='balanced', decision_function_shape='ovr')
                svc.fit(x_train_1, y_train_1)
                y_predict = svc.predict(x_test_1)
                accuracy[i]=accuracy_score(y_predict, y_test_1)
                i=i+1
            avg_accuracy[j,k,l]=np.mean(accuracy)
a=unravel_index(avg_accuracy.argmax(), avg_accuracy.shape)
kernel_final=kernel[a[0]]
C_final=C[a[1]]
gamma_final=gamma[a[2]]
print('kernel_final: ', kernel_final)
print('C_final: ', C_final)
print('gamma_final: ', gamma_final)
print('accuracy: ', np.max(avg_accuracy))

# Training final SVM classifier (with selected features and optimal parameters)
svc_clf = svm.SVC(C=C_final, kernel=kernel_final, gamma=gamma_final, class_weight='balanced', decision_function_shape='ovr')
svc_clf.fit(train_x_sub,train_y)
y_predict_final = svc_clf.predict(test_sub)

Id_id = df_test.columns.get_loc('Id')
ndf = pd.DataFrame(columns=['Id','y'])
ndf['Id'] = df_test.values[:,Id_id]
ndf['y'] = y_predict_final
ndf.info()
#ndf.to_csv('y_pred_file_feature_selection_svm_balanced.csv',index=False)
