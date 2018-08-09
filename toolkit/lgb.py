import numpy as np
import pandas as pd
import lightgbm as lgb
import time


def model_train(lgb_train, num_boost_round):
    params = {
        'learning_rate': 0.03,
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': 'l2',
        'sub_feature': 0.7,
        'bagging_fraction': 0.7,
        'bagging_freq': 1,

        'min_data': 85,
        'max_depth': 14,
        'verbose': -1,
    }

    def fScore(preds, train_data):
        labels = train_data.get_label()
        a = np.log1p(preds) - np.log1p(labels)
        score = np.power(a, 2)
        return 'fScore', score.mean(), False

    gbm = lgb.train(params,
                    lgb_train,
                    feval=fScore,
                    valid_sets=[lgb_train],
                    num_boost_round=num_boost_round,
                    verbose_eval=10, )
    return gbm
