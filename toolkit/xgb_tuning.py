def score(params):
    print("Training with params: ")
    print(params)
    cv_losses=[]
    cv_iteration=[]
    for (train_idx,val_idx) in cv:
        cv_train = X.iloc[train_idx]
        cv_val = X.iloc[val_idx]
        cv_y_train = y[train_idx]
        cv_y_val = y[val_idx]
        
        dtrain = xgb.DMatrix(cv_train,cv_y_train)
        dval = xgb.DMatrix(cv_val,cv_y_val)
        watchlist = [(dtrain, 'train'), (dval, 'valid')]
        
        xgb_model = xgb.train(params, dtrain, 2000, watchlist,
                          verbose_eval=False, 
                          early_stopping_rounds=200)
       
        train_pred = xgb_model.predict(dtrain,ntree_limit=xgb_model.best_ntree_limit)
        val_pred = xgb_model.predict(dval,ntree_limit=xgb_model.best_ntree_limit+1)
        train_loss = root_mean_squared_error(cv_y_train,train_pred)
        val_loss = root_mean_squared_error(cv_y_val,val_pred)
        print('Train RMSE: {},Val RMSE: {}'.format(train_loss,val_loss))
        print('Best iteration: {}'.format(xgb_model.best_ntree_limit))
        cv_losses.append(val_loss)
        cv_iteration.append(xgb_model.best_iteration)
        
        xgb_model.__del__()
    print('6 fold results: {}'.format(cv_losses))
    
    cv_loss_list.append(cv_losses)
    n_iteration_list.append(cv_iteration)
    
    mean_cv_loss = np.mean(cv_losses)
    print('Average iterations: {}'.format(np.mean(cv_iteration)))
    print("Mean Cross Validation RMSE: {}\n".format(mean_cv_loss))
    return {'loss': mean_cv_loss, 'status': STATUS_OK}

def optimize(space,seed=seed,max_evals=5):
    best = fmin(score, space, algo=tpe.suggest, 
        # trials=trials, 
        max_evals=max_evals)
    return best

space = {
    'n_estimators': hp.quniform('n_estimators', 50, 500, 5),
    'max_depth': hp.choice('max_depth', np.arange(5, 10, dtype=int)),
    'subsample': hp.quniform('subsample', 0.7, 0.9, 0.05),
    'colsample_bytree': hp.quniform('colsample_bytree', 0.7, 0.9, 0.05),
    'gamma': hp.quniform('gamma', 0, 1, 0.05),
    'max_leaf_nodes': hp.choice('max_leaf_nodes', np.arange(100,140, dtype=int)),
    'min_child_weight': hp.choice('min_child_weight', np.arange(100,140, dtype=int)),
    'learning_rate': 0.03,
    'eval_metric': 'rmse',
    'objective': 'reg:linear' , 
    'seed': 1204,'tree_method':'gpu_hist'
}

############################  load data ########################
all_data = get_all_data(data_path,'new_sales_lag_after12_month.pickle')
X,y = get_X_y(all_data,33)
X.drop('date_block_num',axis=1,inplace=True)

############################  kfold ###########################
cv = get_cv_idxs(all_data,28,33)

########################  hyperparam search ###################
best_hyperparams = optimize(space,max_evals=200)
print("The best hyperparameters are: ")
print(best_hyperparams)
