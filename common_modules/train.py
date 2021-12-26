import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error as MSE
from sklearn.model_selection import KFold
from lightgbm import LGBMRegressor

from logger import *
from typing import Callable

def fit_lgbm(X, y, params:dict=None, fit_params:dict=None, cv:list=None, metric:Callable=MSE):
    if cv is None:
        cv = KFold(n_splits=5, shuffle=True).split(X)
    
    models = []
    oof_pred = np.zeros_like(y, dtype=np.float64)
    
    for i, (train_idx, valid_idx) in enumerate(cv):
        X_train, y_train = X[train_idx], y[train_idx]
        X_valid, y_valid = X[valid_idx], y[valid_idx]
        
        model = LGBMRegressor(**params)
        with timer(prefix=f"fit fold={i+1}\t", suffix="\n"):
            model.fit(X_train, y_train,
                        eval_set=[(X_train, y_train), (X_valid, y_valid)],
                        **fit_params)
        
        fold_pred = model.predict(X_valid)
        oof_pred[valid_idx] = fold_pred
        
        models.append(model)
        
        fold_score = np.sqrt(metric(y_valid, fold_pred))

        print(f"fold {i+1} score: {fold_score:.2f}")
        print("="*40 + "\n")
        
    oof_score = np.sqrt(metric(y, oof_pred))
                                    
    print(f"FINISHED \ whole score: {oof_score:.2f}")
    print("\n" + "="*40 + "\n")
    return models, oof_pred