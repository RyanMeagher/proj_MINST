from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion, make_pipeline, Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, LabelBinarizer
import numpy as np
import pandas as pd


class Columns(BaseEstimator, TransformerMixin):
    def __init__(self, names=None):
        self.names = names

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X):
        return X[self.names]


def preprocess(X, y, standardization=False, normalization=False):
    y = y.reshape(-1, 1)
    y = OneHotEncoder().fit_transform(y)

    numeric_var = [key for key in dict(X.dtypes)
                   if dict(X.dtypes)[key]
                   in ['float64', 'float32', 'int32', 'int64']]  # Numeric Variable

    cat_var = [key for key in dict(X.dtypes)
               if dict(X.dtypes)[key] in ['object']]  # Categorical Varibles

    if standardization:
        pipe = Pipeline([
            ("features", FeatureUnion([
            ('numeric', make_pipeline(Columns(names=numeric_var), StandardScaler())),
            ('categorical', make_pipeline(Columns(names=cat_var), LabelBinarizer()))]))
        ])
    if normalization:
        pipe = Pipeline([
            ("features", FeatureUnion([
            ('numeric', make_pipeline(Columns(names=numeric_var), MinMaxScaler())),
            ('categorical', make_pipeline(Columns(names=cat_var), LabelBinarizer()))]))
        ])
    return
