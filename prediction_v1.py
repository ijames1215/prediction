import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble.forest import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
from sklearn import preprocessing, linear_model
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score,KFold
from sklearn.svm import SVR
import xgboost as xgb
from sklearn.metrics import mean_squared_error

df = pd.read_csv(r'./prediction_v3.csv')
data = df[['median','average','Follow',"stream_first",'prediction']]
x = data[['median','average','Follow','stream_first','prediction']]
y = x['prediction']

model = LinearRegression(n_jobs = 1)
RF_model = RandomForestRegressor(n_estimators=200, max_features='auto', oob_score=True)
KNN_model = KNeighborsRegressor(n_neighbors=2)
XGBoost_model = xgb.XGBRegressor()
DT_model = DecisionTreeRegressor()#random_state = 0,,max_leaf_nodes = None,max_depth=5

L_rg_list = [];
RF_rg_list = [];
DT_rg_list = [];
KNN_rg_list = [];
SVR_rg_list = [];
XGB_rg_list = []
for i in range(8):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=i)
    # 開始訓練
    model.fit(x_train, y_train)
    RF_model.fit(x_train, y_train)
    KNN_model.fit(x_train, y_train)
    DT_model.fit(x_train, y_train)

    # 開始預測
    predict = model.predict(x_test);
    L_rg_list.append(round(metrics.r2_score(y_test, predict), 3))
    RF_predict = RF_model.predict(x_test);
    RF_rg_list.append(round(metrics.r2_score(y_test, RF_predict), 3))
    KNN_predict = KNN_model.predict(x_test);
    KNN_rg_list.append(round(metrics.r2_score(y_test, KNN_predict), 3))
    DT_predict = DT_model.predict(x_test);
    DT_rg_list.append(round(metrics.r2_score(y_test, DT_predict), 3))

print('R^2 of linear reg: {}, mean: {}'.format(L_rg_list, round(np.mean(L_rg_list), 3)))
print('R^2 of RF reg: {}, mean: {}'.format(RF_rg_list, round(np.mean(RF_rg_list), 3)))
print('R^2 of KNN reg: {}, mean: {}'.format(KNN_rg_list, round(np.mean(KNN_rg_list), 3)))
print('R^2 of DT reg: {}, mean: {}'.format(DT_rg_list, round(np.mean(DT_rg_list), 3)))
print(x_test)
print(y_test)

from joblib import dump,load
dump(model,'./LR_rg_model.sav')
dump(RF_model,'./RF_rg_model.sav')
dump(KNN_model,'./KNN_rg_model.sav')
dump(DT_model,'./DT_rg_model.sav')

loaded_model = load("LR_rg_model.sav")
loaded_RF_model = load("RF_rg_model.sav")

