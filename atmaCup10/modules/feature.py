import pandas as pd
import numpy as np
import texthero as hero
import nltk
import os
import sys
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from gensim.models import word2vec, KeyedVectors

sys.path.append('../..')
from common_modules.feature import *

def input_dir():
    return Path("input")

class NumericBlock(BaseBlock):
    def transform(self, input_df):
        use_cols = [
            'dating_period',
            'dating_year_early',
            'dating_year_late',
        ]

        output_df = input_df[use_cols].copy()
        return output_df

class stringLengthBlock(BaseBlock):
    def __init__(self, col:str):
        self.col = col
        
    def transform(self, input_df):
        output_df = pd.DataFrame()
        col = self.col
        
        output_df[col] = input_df[col].str.len()
        
        return output_df.add_prefix('StringLength_')
    
class AggregationBlock(BaseBlock):
    def __init__(self, table_name:str):
        self.table_name = table_name
    
    def fit(self, input_df):
        _df = pd.read_csv(os.path.join(input_dir() / f"{self.table_name}.csv"))
        
        vc = _df['name'].value_counts()
        _df['CE_name'] = _df['name'].map(vc)
        
        self.agg_df = _df.groupby('object_id')['CE_name'].agg({'max', 'min', 'mean', 'std'})
        
        return self.transform(input_df)
    
    def transform(self, input_df):
        output_df = input_df[['object_id']].merge(self.agg_df, 
                                                  how='left', 
                                                  on='object_id').drop(columns=['object_id'])
        
        return output_df.add_prefix(f'{self.table_name}_')
    
def text_normalization(text):

    # 英語とオランダ語を stopword として指定
    custom_stopwords = nltk.corpus.stopwords.words('dutch') + nltk.corpus.stopwords.words('english')

    x = hero.clean(text, pipeline=[
        hero.preprocessing.fillna,
        hero.preprocessing.lowercase,
        hero.preprocessing.remove_digits,
        hero.preprocessing.remove_punctuation,
        hero.preprocessing.remove_diacritics,
        lambda x: hero.preprocessing.remove_stopwords(x, stopwords=custom_stopwords)
    ])

    return x

class TfidfBlock(BaseBlock):
    """tfidf x SVD による圧縮を行なう block"""
    def __init__(self, column: str):
        """
        args:
            column: str
                変換対象のカラム名
        """
        self.column = column

    def preprocess(self, input_df):
        x = text_normalization(input_df[self.column])
        return x

    def get_master(self, input_df):
        """tdidfを計算するための全体集合を返す. 
        デフォルトでは fit でわたされた dataframe を使うが, もっと別のデータを使うのも考えられる."""
        return input_df

    def fit(self, input_df, y=None):
        master_df = self.get_master(input_df)
        text = self.preprocess(input_df)
        self.pileline_ = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=15_000)),
            ('svd', TruncatedSVD(n_components=100)),
        ])

        self.pileline_.fit(text)
        return self.transform(input_df)

    def transform(self, input_df):
        text = self.preprocess(input_df)
        z = self.pileline_.transform(text)

        out_df = pd.DataFrame(z)
        return out_df.add_prefix(f'{self.column}_tfidf_')
    
class SizeExtractionBlock(BaseBlock):
    def __init__(self, base_df:pd.DataFrame):
        self.base_df = base_df
        
    def fit(self, input_df):
        self.base_df = self.base_df.set_index('object_id')
        
        size_df = self.base_df['sub_title'].str.extractall(r'(h|t|w|l|d) (\d*|\d*\.\d*)(cm|mm|kg)')
        size_df.columns = ['measure', 'num', 'unit']
        
        size_df['num'] = size_df['num'].astype(float)
        size_df['num'] = size_df.apply(lambda row: row['num'] * 10 if row['unit']=='cm' else row['num'], axis=1)
        
        size_df['unit'] = size_df['unit'].replace({'cm':'mm'})
        
        size_df['measure'] = size_df['measure'] + '_' + size_df['unit']
        
        size_df.drop(columns=['unit'], inplace=True)
        size_df.reset_index(inplace=True)
        size_df = size_df.rename({'level_0':'object_id'})
        
        meta_df = pd.DataFrame()
        
        for g, group in size_df.groupby('object_id'):
            _df = pd.pivot_table(data=group, values='num', index='object_id', columns='measure', aggfunc=np.sum)
            meta_df = pd.concat([meta_df, _df], axis=0)
        
        meta_df = meta_df.reset_index()
        self.meta_df = meta_df
        
        return self.transform(input_df)
    
    def transform(self, input_df):
        output_df = input_df[['object_id']].merge(self.meta_df, how='left', on='object_id').drop(columns=['object_id'])

        return output_df

class Word2vecBlock(BaseBlock):
    def __init__(self, base_df:pd.DataFrame, col:str, size:int=50, n_iter:int=500):
        self.col = col
        self.base_df = base_df
        self.size = size
        self.n_iter = n_iter
        
    def fit(self, input_df):
        w2v_model = word2vec.Word2Vec(self.base_df[self.col].fillna(''), 
                                      size=self.size,
                                      min_count=1, 
                                      iter=self.n_iter)
        self.w2v_model = w2v_model
        return self.transform(input_df)
    
    def transform(self, input_df):
        vectors = input_df[self.col].fillna('').apply(lambda row:np.mean([self.w2v_model.wv[e] for e in row], axis=0))
        
        output_df = pd.DataFrame.from_dict(vectors.to_dict()).T
        
        return output_df.add_prefix(f'w2v_{self.col}_')

class RelationCountBlock(BaseBlock):
    def __init__(self, name:str):
        self.name = name
        
    def fit(self, input_df):
        other_df = read_csv(self.name)
        self.agg_df = other_df.groupby('object_id').size().rename('size').reset_index()
        return self.transform(input_df)
    
    def transform(self, input_df):
        output_df = pd.merge(input_df['object_id'], self.agg_df, how='left', on='object_id').drop(columns=['object_id'])
        output_df = output_df.fillna(0).astype(int)
        return output_df.add_prefix(f'Counts_{self.name}_')

class One2ManyBlock(BaseBlock):
    def __init__(self, name):
        self.many_df = read_csv(name)
        self.name = name
        
    def fit(self, input_df):
        minimum_freq = 30
        vc = self.many_df['name'].value_counts()
        vc = vc[vc > minimum_freq]
        use_df = self.many_df[self.many_df['name'].isin(vc.index)].reset_index(drop=True)
        
        self.agg_df = pd.crosstab(index=use_df['object_id'], columns=use_df['name']).reset_index()
        return self.transform(input_df)
    
    def transform(self, input_df):
        output_df = pd.merge(input_df['object_id'], self.agg_df, how='left', on='object_id').drop(columns=['object_id'])
        output_df = output_df.fillna(0).astype(int)
        output_df['total_occurance'] = output_df.sum(axis=1)
        return output_df.add_prefix(self.name + '__')