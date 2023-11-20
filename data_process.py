import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from utils import scale_data, inverse_scale_data

class DataProcessor:
    def __init__(self, path):
        self.data_path = path
        self.original_data = None
        self.data = None
        self.synthetic_data = None

    def load_and_preprocess(self, num_columns):
        self.data = pd.read_csv(self.data_path, sep='\t', index_col=0).T
        self.reduce_columns(num_columns)
        self.original_data = self.data
        self.data = scale_data(self.data)

    def load_and_postprocess(self, synthetic_data):
        self.synthetic_data = synthetic_data
        original_min, original_max = self.original_data.min().min(), self.original_data.max().max()
        self.synthetic_data = inverse_scale_data(self.synthetic_data, original_min, original_max)

    def save_generated_data(self, path):
        df = pd.DataFrame(self.synthetic_data)
        df.columns = self.original_data.columns
        df.to_csv(path, sep='\t', index=False, header=True)

    def reduce_columns(self, num_columns):
        selector = VarianceThreshold(threshold=0)
        selector.fit(self.data)
        variances = selector.variances_
        sorted_variances_indices = np.argsort(variances)[-num_columns:]
        self.data = self.data.iloc[:, sorted_variances_indices]

    def get_data(self):
        return self.data
    
    def get_original_data(self):
        return self.original_data

    def get_data_shape(self):
        return self.data.shape[1]
