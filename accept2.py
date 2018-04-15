
import pandas as pd
import numpy as np

from sklearn import datasets
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

df1 = pd.read_csv('/case2_training.csv')
df2 = pd.read_csv('/case2_testing.csv')

df11 = df1.loc[df1.Accept==1,]
df12 = df1.loc[df1.Accept==0,]

len(df11)
# df13 = df12.sample(n=len(df11),random_state=0)
# df14 = pd.concat([df11,df13])

X1 = df1.loc[:,('Date','Apartment','Beds','Review','Pic Quality','Price')]
X2 = pd.DataFrame(df1.loc[:,('Region','Weekday')].astype(str))
X3 = pd.get_dummies(X2)
y = df1.Accept
d = pd.concat([X1,X3],axis=1)
all_df = pd.DataFrame(pd.concat([d,y],axis=1))
X = all_df.drop(['Accept'],axis=1)

from sklearn.preprocessing import scale
# X = scale(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
print(len(y_test),sum(y_test))

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train, y_train) #拟合

predict_prob = lr.predict_proba(X_test)
predict_data = lr.predict(X_test)
res_df = pd.DataFrame({'y': y_test,'pred':predict_data})
res_df['acc'] = res_df.pred - res_df.y
accuracy = len(res_df.loc[res_df.acc==0,])/len(res_df)
print(accuracy)

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, predict_data)
print (confusion_matrix)


#画图roc曲线
from sklearn.metrics import roc_curve, auc
fpr, tpr, thresholds = roc_curve(y_test, predict_prob[:, 1])
roc_auc = auc(fpr, tpr)
print(roc_auc)
plt.plot(fpr, tpr, lw=1, label='ROC fold  (area = %0.2f)' % (roc_auc))
plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label=roc_auc)
plt.legend(loc='best')

import xgboost as xgb
dtrain = xgb.DMatrix(X_train, label = y_train)
dtest = xgb.DMatrix(X_test)

params={'booster':'gbtree',
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'max_depth':3,
    # 'lambda':10,
    'subsample':0.8,
    'colsample_bytree':0.65,
    #'min_child_weight':2,
    'eta': 0.025,
    'seed':0,
    #'nthread':8,
     'silent':1}

watchlist = [(dtrain, 'train')]
bst = xgb.train(params, dtrain, num_boost_round = 1000, evals = watchlist)
ypred = bst.predict(dtest)

# 设置阈值, 输出一些评价指标
y_pred = (ypred >= 0.5) * 1

from sklearn import metrics
print ('AUC: %.4f' % metrics.roc_auc_score(y_test, ypred))
print ('ACC: %.4f' % metrics.accuracy_score(y_test, y_pred))
print ('Recall: %.4f' % metrics.recall_score(y_test, y_pred))
print ('F1-score: %.4f' % metrics.f1_score(y_test, y_pred))
print ('Precesion: %.4f' % metrics.precision_score(y_test, y_pred))
print(metrics.confusion_matrix(y_test,y_pred))


#结果预测
#预测数据处理
X21 = df2.loc[:,('Date','Apartment','Beds','Review','Pic Quality','Price')]
X22 = pd.DataFrame(df2.loc[:,('Region','Weekday')].astype(str))
X23 = pd.get_dummies(X22)

d2 = pd.concat([X21,X23],axis=1)
# all_df2 = pd.DataFrame(pd.concat([d2,y2],axis=1))
all_df2 = pd.DataFrame(d2)
# X2 = all_df2.drop(['Accept'],axis=1)
future_pred_prob = lr.predict_proba(all_df2)
df2['prob'] = [x[1] for x in future_pred_prob]
df3 = df2[['ID','prob']]
df3.tail(1)
df3.to_csv('E:/work_dir/tasks/task22/lajiduanxin/case2_future_prob.csv',index=False)