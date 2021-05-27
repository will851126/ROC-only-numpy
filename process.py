from collections import defaultdict
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
import numpy as np

class Preprocess(BaseEstimator, TransformerMixin):

    def __init__(self, num_cols = None, cat_cols = None):
        self.num_cols = num_cols
        self.cat_cols = cat_cols

    def fit(self, data):
        """
        Fit the Preprocess Transformer

        Parameters
        ----------
        data : DataFrame
        """
        data = data.copy()

        # Label encoding across multiple columns in scikit-learn
        # https://stackoverflow.com/questions/24458645/label-encoding-across-multiple-columns-in-scikit-learn
        if self.cat_cols is not None:
            self.label_encode_dict_ = defaultdict(LabelEncoder)
            label_encoded = (data[self.cat_cols].
                             apply(lambda x: self.label_encode_dict_[x.name].fit_transform(x)))

            self.cat_encode_ = OneHotEncoder(sparse = False)
            self.cat_encode_.fit(label_encoded)

        if self.num_cols is not None:
            self.scaler_ = StandardScaler().fit(data[self.num_cols])

        # store the column names (numeric columns comes before the
        # categorical columns) so we can refer to them later
        if self.num_cols is not None:
            colnames = self.num_cols.copy()
        else:
            colnames = []

        if self.cat_cols is not None:
            for col in self.cat_cols:
                cat_colnames = [col + '_' + str(classes)
                                for classes in self.label_encode_dict_[col].classes_]
                colnames += cat_colnames

        self.colnames_ = colnames
        return self

    def transform(self, data):
        """
        Trasform the data using the fitted Preprocess Transformer

        Parameters
        ----------
        data : DataFrame
        """
        if self.cat_cols is not None:
            label_encoded = (data[self.cat_cols].
                             apply(lambda x: self.label_encode_dict_[x.name].transform(x)))
            cat_encoded = self.cat_encode_.transform(label_encoded)

        if self.num_cols is not None:
            scaled = self.scaler_.transform(data[self.num_cols])

        # combine encoded categorical columns and scaled numerical
        # columns, it's the same as concatenate it along axis 1
        if self.cat_cols is not None and self.num_cols is not None:
            X = np.hstack((scaled, cat_encoded))
        elif self.num_cols is None:
            X = cat_encoded
        else:
            X = scaled

        return X