import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error as MSE, mean_squared_log_error as MSLE
from lightgbm import LGBMRegressor

import sys
sys.path.append('../..')
from common_modules.util import *

def fit_lgbm(X, y, params:dict=None, fit_params:dict=None, cv:list=None):
    if cv is None:
        cv = KFold(n_splits=5, shuffle=True).split(X)
    
    models = []
    oof_pred = np.zeros_like(y, dtype=np.float64)
    
    y_d = np.log1p(y)
    
    for i, (train_idx, valid_idx) in enumerate(cv):
        X_train, y_train = X[train_idx], y_d[train_idx]
        X_valid, y_valid = X[valid_idx], y_d[valid_idx]
        
        model = LGBMRegressor(**params)
        with timer(prefix=f"fit fold={i+1}\t", suffix="\n"):
            model.fit(X_train, y_train,
                        eval_set=[(X_train, y_train), (X_valid, y_valid)],
                        **fit_params)
        
        fold_pred_d = model.predict(X_valid)
        fold_pred = np.expm1(fold_pred_d)
        fold_pred = np.where(fold_pred < 0 , 0, fold_pred)
        oof_pred[valid_idx] = fold_pred
        
        models.append(model)
        
        fold_score = np.sqrt(MSE(y_valid, fold_pred_d))

        print(f"fold {i+1} RMSLE: {fold_score:.2f}")
        print("="*40 + "\n")
        
    oof_score = np.sqrt(MSLE(y, oof_pred))
                                    
    print(f"FINISHED \ whole score: {oof_score:.2f}")
    print("\n" + "="*40 + "\n")
    return models, oof_pred