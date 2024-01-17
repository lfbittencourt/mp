import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import make_column_transformer
from sklearn.pipeline import FeatureUnion, _name_estimators


class ColumnTransformerUnion(FeatureUnion):
    def __init__(self, transformers):
        self.transformers = transformers

        super().__init__(
            _name_estimators(
                [
                    make_column_transformer(transformer, remainder="drop")
                    for transformer in transformers
                ]
            )
        )


class ListColumnExpander(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if X.ndim == 1:
            return pd.DataFrame(X.to_list())

        return pd.DataFrame(X.iloc[:, 0].to_list())
