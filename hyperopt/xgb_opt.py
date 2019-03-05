#coding:utf8
'''
建议使用python2.7 
pip install xgboost
pip install hyperopt

step1: 定义参数空间 
step2: 定义评估函数
step3: 训练获取最优参数模型
step4: 根据最优参数训练模型
目前 hyperopt 的优化算法所识别的随机表达式是：
hp.choice(label, options)
返回其中一个选项，它应该是一个列表或元组。options元素本身可以是[嵌套]随机表达式。在这种情况下，仅出现在某些选项中的随机选择(stochastic choices)将成为条件参数。

hp.randint(label, upper)
返回范围:[0，upper]中的随机整数。当更远的整数值相比较时,这种分布的语义是意味着邻整数值之间的损失函数没有更多的相关性。例如，这是描述随机种子的适当分布。如果损失函数可能更多的与相邻整数值相关联，那么你或许应该用“量化”连续分布的一个，比如 quniform ， qloguniform ， qnormal 或 qlognormal 。

hp.uniform(label, low, high)
返回位于[low,hight]之间的均匀分布的值。
在优化时，这个变量被限制为一个双边区间。

hp.quniform(label, low, high, q)
返回一个值，如 round（uniform（low，high）/ q）* q
适用于目标仍然有点“光滑”的离散值，但是在它上下存在边界(双边区间)。

hp.loguniform(label, low, high)
返回根据 exp（uniform（low，high）） 绘制的值，以便返回值的对数是均匀分布的。 优化时，该变量被限制在[exp（low），exp（high）]区间内。

hp.qloguniform(label, low, high, q)
返回类似 round（exp（uniform（low，high））/ q）* q 的值
适用于一个离散变量，其目标是“平滑”，并随着值的大小变得更平滑，但是在它上下存在边界(双边区间)。

hp.normal(label, mu, sigma)
返回正态分布的实数值，其平均值为 mu ，标准偏差为 σ。优化时，这是一个无约束(unconstrained)的变量。

hp.qnormal(label, mu, sigma, q)
返回像 round（正常（mu，sigma）/ q）* q 的值
适用于离散值，可能需要在 mu 附近的取值，但从基本上上是无约束(unbounded)的。

hp.lognormal(label, mu, sigma)(对数正态分布)
返回根据 exp（normal（mu，sigma）） 绘制的值，以便返回值的对数正态分布。优化时，这个变量被限制为正值。

hp.qlognormal(label, mu, sigma, q)
返回类似 round（exp（normal（mu，sigma））/ q）* q 的值
适用于一个离散变量，其目标是“平滑”，并随着值的大小变得更平滑，变量的大小是从一个边界开始的(单边区间)。
'''
import sys
from hyperopt import fmin, tpe, hp, partial
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import pandas as pd

def GetNewDataByPandas():
    wine = pd.read_csv("./wine.csv", sep=';')
    wine['alcohol**2'] = pow(wine["alcohol"], 2)
    wine['volatileAcidity*alcohol'] = wine["alcohol"] * wine['volatile acidity']
    print(wine.isnull().sum())

    y = np.array(wine.quality)
    X = np.array(wine.drop("quality", axis=1))
    columns = np.array(wine.columns)
    return X, y, columns

x,y,columns = GetNewDataByPandas()

# split data to [[0.8,0.2],01]
x_train_all, x_predict, y_train_all, y_predict = train_test_split(x, y, test_size=0.10, random_state=100)
x_train, x_test, y_train, y_test = train_test_split(x_train_all, y_train_all, test_size=0.2, random_state=100)

dtrain = xgb.DMatrix(data=x_train,label=y_train,missing=-999.0)
dtest = xgb.DMatrix(data=x_test,label=y_test,missing=-999.0)

evallist = [(dtest, 'eval'), (dtrain, 'train')]	


# 自定义hyperopt的参数空间
space = {"max_depth": hp.randint("max_depth", 15),
         "n_estimators": hp.randint("n_estimators", 300),
         'learning_rate': hp.uniform('learning_rate', 1e-3, 5e-1),
         "subsample": hp.randint("subsample", 5),
         "min_child_weight": hp.randint("min_child_weight", 6),
         }

def argsDict_tranform(argsDict, isPrint=False):
    argsDict["max_depth"] = argsDict["max_depth"] + 5
    argsDict['n_estimators'] = argsDict['n_estimators'] + 150
    argsDict["learning_rate"] = argsDict["learning_rate"] * 0.02 + 0.05
    argsDict["subsample"] = argsDict["subsample"] * 0.1 + 0.5
    argsDict["min_child_weight"] = argsDict["min_child_weight"] + 1
    if isPrint:
        print(argsDict)
    else:
        pass

    return argsDict
def xgboost_factory(argsDict):
    argsDict = argsDict_tranform(argsDict)

    params = {'nthread': -1,  # 进程数
              'max_depth': argsDict['max_depth'],  # 最大深度
              'n_estimators': argsDict['n_estimators'],  # 树的数量
              'eta': argsDict['learning_rate'],  # 学习率
              'subsample': argsDict['subsample'],  # 采样数
              'min_child_weight': argsDict['min_child_weight'],  # 终点节点最小样本占比的和
              'objective': 'reg:linear',
              'silent': 0,  # 是否显示
              'gamma': 0,  # 是否后剪枝
              'colsample_bytree': 0.7,  # 样本列采样
              'alpha': 0,  # L1 正则化
              'lambda': 0,  # L2 正则化
              'scale_pos_weight': 0,  # 取值>0时,在数据不平衡时有助于收敛
              'seed': 100,  # 随机种子
              'missing': -999,  # 填充缺失值
              }
    params['eval_metric'] = ['rmse']
    xrf = xgb.train(params, dtrain, 300, evallist,early_stopping_rounds=100)
    return get_tranformer_score(xrf)

def get_tranformer_score(tranformer):
    xrf = tranformer
    dpredict = xgb.DMatrix(x_predict)
    prediction = xrf.predict(dpredict, ntree_limit=xrf.best_ntree_limit)
    return mean_squared_error(y_predict, prediction)

# 开始使用hyperopt进行自动调参
algo = partial(tpe.suggest, n_startup_jobs=1)
best = fmin(xgboost_factory, space, algo=algo, max_evals=20, pass_expr_memo_ctrl=None)

RMSE = xgboost_factory(best)
print('best :', best)
print('best param after transform :')
argsDict_tranform(best,isPrint=True)
print('rmse of the best xgboost:', np.sqrt(RMSE))
