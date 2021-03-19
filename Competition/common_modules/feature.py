import pandas as pd
import numpy as np
from tqdm.notebook import tqdm

from common_modules.util import *

class BaseBlock(object):
    def fit(self, input_df, y=None):
        return self.transform(input_df)
    
    def transform(self, input_df):
        return NotImplementedError()
    
class CountEncoding(BaseBlock):
    def __init__(self, col:str, base_df):
        self.col = col
        self.base_df = base_df
        
    def transform(self, input_df):
        output_df = pd.DataFrame()
        col = self.col

        vc = self.base_df[col].value_counts()
        output_df[col] = input_df[col].map(vc)
        
        return output_df.add_prefix(f"CE_")
    
class TargetEncoding(BaseBlock):
    def __init__(self, col:str, target_col:str, cv:list, weight:int=100):
        self.col = col
        self.target_col = target_col
        self.cv = list(cv)
        self.weight = weight

    def fit(self, input_df):
        X = input_df[self.col]
        y = input_df[self.target_col]
        
        output_df = pd.DataFrame()
        oof = np.zeros_like(y, dtype=np.float)
        
        for train_idx, valid_idx in self.cv:
            y_mean = y[train_idx].mean()
            agg_df = y[train_idx].groupby(X[train_idx]).agg(['count', 'mean'])
            counts = agg_df['count']
            group_mean = agg_df['mean']
            weight = self.weight

            _df = (counts * group_mean + weight * y_mean) / (counts + weight)
            _df = _df.reindex(X.unique())
            _df = _df.fillna(_df.mean())
            oof[valid_idx] = X[valid_idx].map(_df)
            
        output_df[self.target_col] = oof
        self.meta_df = y.groupby(X).mean()
        return output_df.add_prefix(f"TE@{self.col}=")
    
    def transform(self, input_df):
        output_df = pd.merge(input_df[self.col], self.meta_df, how='left', on=self.col).drop(columns=[self.col])
        return output_df.add_prefix(f"TE@{self.col}=")
    
class OneHotEncoding(BaseBlock):
    def __init__(self, col:str, threshold:int=40):
        self.col = col
        self.threshold = threshold
        
    def fit(self, input_df):
        vc = input_df[self.col].dropna().value_counts()
        cats = vc[vc > self.threshold].index
        self.cats = cats
        return self.transform(input_df)
    
    def transform(self, input_df):
        x = pd.Categorical(input_df[self.col], categories=self.cats)
        output_df = pd.get_dummies(x, dummy_na=False)
        output_df.columns = output_df.columns.tolist()
        
        return output_df.add_prefix(f'OH@{self.col}=')
    
class WrapperBlock(BaseBlock):
    def __init__(self, function):
        self.function = function

    def transform(self, input_df):
        return self.function(input_df)
    
def get_function(block, is_train):
  s = mapping = {
      True: 'fit',
      False: 'transform'
  }.get(is_train)
  return getattr(block, s)

def to_features(input_df, blocks, is_train=False):
    output_df = pd.DataFrame()

    for block in tqdm(blocks, total=len(blocks)):
        func = get_function(block, is_train)
        prefix = 'create ' + block.__class__.__name__ + ' ' + func.__name__ + ' '

        with timer(prefix=prefix):
            _df = func(input_df)

        assert len(_df) == len(input_df), block
        output_df = pd.concat([output_df, _df], axis=1)

    return output_df