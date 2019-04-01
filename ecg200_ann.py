import numpy as np
import math
import os
import time
from pathlib import PurePath
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import filtfilt, butter, decimate, find_peaks
import shutil
import glob
from matplotlib.ticker import NullLocator
from pandas import DataFrame, Series
import tensorflow as tf
from keras.layers import Dense, Input, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.models import Model
from keras.callbacks import EarlyStopping
from arff2pandas import a2p
from sklearn import preprocessing
from pandas import DataFrame
import pandas as pd
from keras.callbacks import Callback


class PreprocessFeature():
    '''
        Parent class for normalize and standardize classes.
        Provide some helper methods
    '''

    def __init__(self, dataframe, features):
        self._df = dataframe
        self._features = features

    def show_train_feature_distribution(self, feature):
        # make bins intervals
        plt.hist(self._df[feature].values, bins='auto')  # arguments are passed to np.histogram
        plt.title("Histogram with 'auto' bins")
        plt.show()

    def show_test_feature_distribution(test_df, feature):
        # make bins intervals
        plt.hist(test_df[feature].values, bins='auto')  # arguments are passed to np.histogram
        plt.title("Histogram with 'auto' bins")
        plt.show()


class NormalizeFeatures(PreprocessFeature):
    '''
        Class to normalize input features in dataframe. Each feature in column
        Return normalized dataframe and norms-vector.
        Norm vector used for future input data.
        Future data ougth to normalzied according to train dataset statistics.
    '''

    def __init__(self, dataframe, features):
        '''
            dataframe - input dataframe with features as columns
            features - list of features to normalize
            normalized_df - normalized dataframe to return
            norms_df - dataframe with norms for future usage
        '''
        super().__init__(dataframe, features)
        self._normalized_df = None
        self._norms_df = None

    def show_normalize_feature_distribution(self, feature):
        # make bins intervals
        plt.hist(self._normalized_df[feature].values, bins='auto')  # arguments are passed to np.histogram
        plt.title("Histogram with 'auto' bins")
        plt.show()

    @staticmethod
    def normalize(vector):
        vector_min = np.min(vector)
        vector_max = np.max(vector)
        delta = vector_max - vector_min
        if delta != 0:
            normalized_vector = np.array([(x - vector_min) / delta for x in vector])
        else:
            raise Exception(ZeroDivisionError)
        return normalized_vector, vector_min, vector_max

    def normalize_train_dataframe(self):
        self._normalized_df = DataFrame()
        self._norms_df = DataFrame()
        for col in self._features:
            normalized_vector, vector_min, vector_max = self.normalize(pd.to_numeric(self._df[col], errors='raise'))
            self._normalized_df.insert(loc=len(self._normalized_df.columns), column=col, value=normalized_vector)
            self._norms_df.insert(loc=len(self._norms_df.columns), column=col, value=(vector_min, vector_max))
            self._norms_df.rename(index={0: 'feature_min', 1: 'feature_max'}, inplace=True)

    def get_normalized_train_df(self):
        if self._normalized_df.empty is False:
            return self._normalized_df
        else:
            raise Exception('Normalized dataframe is empty')

    def get_norms_df(self):
        if self._norms_df.empty is False:
            return self._norms_df
        else:
            raise Exception('Normalized dataframe is empty')

    def load_norms_df(self, norm_df):
        self._norms_df = norm_df

    def normalize_test_series(self, test_series):
        # numeraized_test.series = pd.to_numeric(self._df[col], errors='raise')
        normalized_vector_list = []
        for ind in test_series.index:
            # print(type(test_series[ind]))
            # print(type(self._norms_df[ind]['feature_min']))
            delta = self._norms_df[ind]['feature_max'] - self._norms_df[ind]['feature_min']
            normalized_value = (float(test_series[ind]) - self._norms_df[ind]['feature_min']) / delta
            normalized_vector_list.append(normalized_value)
        normalized_test_series = Series(data=normalized_vector_list, index=test_series.index)
        return normalized_test_series

    def normalize_test_dataframe(self, test_df):
        normalized_test_df = DataFrame()
        for index, row in test_df.iterrows():
            # print(row.index)
            normalized_series = self.normalize_test_series(row)
            # print(normalized_series)
            normalized_test_df = normalized_test_df.append(normalized_series, ignore_index=True)
        # print(normalized_test_df)
        return normalized_test_df


class StandardizeFeatures(PreprocessFeature):
    '''
        Class to standardize input features in dataframe. Each feature vector in column.
        Return standardized dataframe and standardize vector with mean and std for each feature.
        Standardize vector used for future input data.
        Future data ougth to standardized according to train dataset statistics.
    '''

    def __init__(self, dataframe, features):
        '''
            dataframe - input dataframe with features as columns
            features - list of features to normalize
            standardized_df - standardized dataframe to return
            mean_std_df - dataframe with mean and std for future usage
        '''
        super().__init__(dataframe, features)
        self._standardized_df = None
        self._mean_std_df = None

    def show_standardize_feature_distribution(self, feature):
        # make bins intervals
        plt.hist(self._stn_df[feature].values, bins='auto')  # arguments are passed to np.histogram
        plt.title("Histogram with 'auto' bins")
        plt.show()

    @staticmethod
    def standardize(vector):
        vector_mean = np.mean(vector)
        vector_std = np.std(vector)
        if vector_std != 0:
            standardized_vector = np.array([(x - vector_mean) / vector_std for x in vector])
        else:
            raise Exception(ZeroDivisionError)
        return standardized_vector, vector_mean, vector_std

    def standardize_train_dataframe(self):
        self._standardized_df = DataFrame()
        self._mean_std_df = DataFrame()
        for col in self._features:
            standardized_vector, vector_mean, vector_std = self.standardize(pd.to_numeric(self._df[col],
                                                                                          errors='raise'))
            self._standardized_df.insert(loc=len(self._standardized_df.columns), column=col, value=standardized_vector)
            self._mean_std_df.insert(loc=len(self._mean_std_df.columns), column=col, value=(vector_mean, vector_std))
            self._mean_std_df.rename(index={0: 'feature_mean', 1: 'feature_std'}, inplace=True)

    def get_standardized_train_df(self):
        if self._standardized_df.empty is False:
            return self._standardized_df
        else:
            raise Exception('Standardized dataframe is empty')

    def get_mean_std_df(self):
        if self._mean_std_df.empty is False:
            return self._mean_std_df
        else:
            raise Exception('Standardized dataframe is empty')

    def load_mean_std_df(self, mean_std_df):
        self._mean_std_df = mean_std_df

    def standardize_test_series(self, test_series):
        standardized_vector_list = []
        for ind in test_series.index:
            standardized_value = ((float(test_series[ind]) - self._mean_std_df[ind]['feature_mean']) /
                                  self._mean_std_df[ind]['feature_std'])
            standardized_vector_list.append(standardized_value)
        standardized_test_series = Series(data=standardized_vector_list, index=test_series.index)
        return standardized_test_series

    def standardize_test_dataframe(self, test_df):
        standardized_test_df = DataFrame()
        for index, row in test_df.iterrows():
            # print(row.index)
            standardized_series = self.standardize_test_series(row)
            # print(normalized_series)
            standardized_test_df = standardized_test_df.append(standardized_series, ignore_index=True)
        return standardized_test_df


class StNFeatures(PreprocessFeature):
    '''
        Class to standardize input features in dataframe. Each feature vector in column.
        Return standardized dataframe and standardize vector with mean and std for each feature.
        Standardize vector used for future input data.
        Future data ougth to standardized according to train dataset statistics.
    '''

    def __init__(self, dataframe, features):
        '''
            dataframe - input dataframe with features as columns
            features - list of features to normalize
            standardized_df - standardized dataframe to return
            mean_std_df - dataframe with mean and std for future usage
        '''
        super().__init__(dataframe, features)
        self._stn_df = None
        self._mean_std_df = None
        self._norm_df = None

    def show_stn_feature_distribution(self, feature):
        # make bins intervals
        plt.hist(self._standardized_df[feature].values, bins='auto')  # arguments are passed to np.histogram
        plt.title("Histogram with 'auto' bins")
        plt.show()

    @staticmethod
    def standardize_normalize(vector):
        vector_mean = np.mean(vector)
        vector_std = np.std(vector)
        if vector_std != 0:
            standardized_vector = np.array([(x - vector_mean) / vector_std for x in vector])
        else:
            raise Exception(ZeroDivisionError)
        standardized_vector_min = np.min(standardized_vector)
        standardized_vector_max = np.max(standardized_vector)
        delta = standardized_vector_max - standardized_vector_min
        if delta != 0:
            stn_vector = np.array([(x - standardized_vector_min) / delta for x in standardized_vector])
        else:
            raise Exception(ZeroDivisionError)

        return stn_vector, vector_mean, vector_std, standardized_vector_min, standardized_vector_max

    def stn_train_dataframe(self):
        self._stn_df = DataFrame()
        self._mean_std_df = DataFrame()
        self._norm_df = DataFrame()
        for col in self._features:
            (stn_vector, vector_mean,
             vector_std, stn_min, stn_max) = self.standardize_normalize(pd.to_numeric(self._df[col], errors='raise'))
            self._stn_df.insert(loc=len(self._stn_df.columns), column=col, value=stn_vector)
            self._mean_std_df.insert(loc=len(self._mean_std_df.columns), column=col, value=(vector_mean, vector_std))
            self._mean_std_df.rename(index={0: 'feature_mean', 1: 'feature_std'}, inplace=True)
            self._norm_df.insert(loc=len(self._norm_df.columns), column=col, value=(stn_min, stn_max))
            self._norm_df.rename(index={0: 'feature_min', 1: 'feature_max'}, inplace=True)

    def get_stn_train_df(self):
        if self._stn_df.empty is False:
            return self._stn_df
        else:
            raise Exception('Standardized dataframe is empty')

    def get_mean_std_df(self):
        if self._mean_std_df.empty is False:
            return self._mean_std_df
        else:
            raise Exception('Standardized dataframe is empty')

    def get_norm_df(self):
        if self._norm_df.empty is False:
            return self._norm_df
        else:
            raise Exception('Standardized dataframe is empty')

    def load_mean_std_df(self, mean_std_df):
        self._mean_std_df = mean_std_df

    def load_norm_df(self, norm_df):
        self._norm_df = norm_df

    def stn_test_series(self, test_series):
        stn_vector_list = []
        for ind in test_series.index:
            standardized_value = ((float(test_series[ind]) - self._mean_std_df[ind]['feature_mean']) /
                                  self._mean_std_df[ind]['feature_std'])
            delta = self._norm_df[ind]['feature_max'] - self._norm_df[ind]['feature_min']
            stn_value = (standardized_value - self._norm_df[ind]['feature_min']) / delta
            stn_vector_list.append(stn_value)
        stn_test_series = Series(data=stn_vector_list, index=test_series.index)
        return stn_test_series

    def stn_test_dataframe(self, test_df):
        stn_test_df = DataFrame()
        for index, row in test_df.iterrows():
            # print(row.index)
            stn_series = self.stn_test_series(row)
            # print(normalized_series)
            stn_test_df = stn_test_df.append(stn_series, ignore_index=True)
        return stn_test_df


class NStFeatures(PreprocessFeature):
    '''
        Class to standardize input features in dataframe. Each feature vector in column.
        Return standardized dataframe and standardize vector with mean and std for each feature.
        Standardize vector used for future input data.
        Future data ougth to standardized according to train dataset statistics.
    '''

    def __init__(self, dataframe, features):
        '''
            dataframe - input dataframe with features as columns
            features - list of features to normalize
            standardized_df - standardized dataframe to return
            mean_std_df - dataframe with mean and std for future usage
        '''
        super().__init__(dataframe, features)
        self._nst_df = None
        self._norm_df = None
        self._mean_nst_df = None

    def show_nst_feature_distribution(self, feature):
        # make bins intervals
        plt.hist(self._nst_df[feature].values, bins='auto')  # arguments are passed to np.histogram
        plt.title("Histogram with 'auto' bins")
        plt.show()

    @staticmethod
    def normalize_standardize(vector):
        vector_min = np.min(vector)
        vector_max = np.max(vector)
        delta = vector_max - vector_min
        if delta != 0:
            normalized_vector = np.array([(x - vector_min) / delta for x in vector])
        else:
            raise Exception(ZeroDivisionError)
        normalized_vector_mean = np.mean(normalized_vector)
        normalized_vector_std = np.std(normalized_vector)
        if normalized_vector_std != 0:
            nst_vector = np.array([(x - normalized_vector_mean) / normalized_vector_std for x in normalized_vector])
        else:
            raise Exception(ZeroDivisionError)
        return nst_vector, vector_min, vector_max, normalized_vector_mean, normalized_vector_std

    def nst_train_dataframe(self):
        self._nst_df = DataFrame()
        self._norm_df = DataFrame()
        self._mean_nst_df = DataFrame()
        for col in self._features:
            (nst_vector, vector_min,
             vector_max, nst_mean, nst_std) = self.normalize_standardize(pd.to_numeric(self._df[col], errors='raise'))
            self._nst_df.insert(loc=len(self._nst_df.columns), column=col, value=nst_vector)
            self._norm_df.insert(loc=len(self._norm_df.columns), column=col, value=(vector_min, vector_max))
            self._norm_df.rename(index={0: 'feature_min', 1: 'feature_max'}, inplace=True)
            self._mean_nst_df.insert(loc=len(self._mean_nst_df.columns), column=col, value=(nst_mean, nst_std))
            self._mean_nst_df.rename(index={0: 'feature_mean', 1: 'feature_std'}, inplace=True)

    def get_nst_train_df(self):
        if self._nst_df.empty is False:
            return self._nst_df
        else:
            raise Exception('Standardized dataframe is empty')

    def get_mean_nst_df(self):
        if self._mean_nst_df.empty is False:
            return self._mean_nst_df
        else:
            raise Exception('Standardized dataframe is empty')

    def get_norm_df(self):
        if self._norm_df.empty is False:
            return self._norm_df
        else:
            raise Exception('Standardized dataframe is empty')

    def load_mean_nst_df(self, mean_nst_df):
        self._mean_nst_df = mean_nst_df

    def load_norm_df(self, norm_df):
        self._norm_df = norm_df

    def nst_test_series(self, test_series):
        nst_vector_list = []
        for ind in test_series.index:
            delta = self._norm_df[ind]['feature_max'] - self._norm_df[ind]['feature_min']
            normalized_value = ((float(test_series[ind]) - self._norm_df[ind]['feature_min']) / delta)
            nst_value = ((normalized_value - self._mean_nst_df[ind]['feature_mean']) /
                         self._mean_nst_df[ind]['feature_std'])
            nst_vector_list.append(nst_value)
        nst_test_series = Series(data=nst_vector_list, index=test_series.index)
        return nst_test_series

    def nst_test_dataframe(self, test_df):
        nst_test_df = DataFrame()
        for index, row in test_df.iterrows():
            # print(row.index)
            nst_series = self.nst_test_series(row)
            # print(normalized_series)
            nst_test_df = nst_test_df.append(nst_series, ignore_index=True)
        return nst_test_df


class TerminateOnBaseline(Callback):
    """Callback that terminates training when either acc or val_acc reaches a specified baseline
    """

    def __init__(self, monitor='acc', baseline=0.99):
        super(TerminateOnBaseline, self).__init__()
        self.monitor = monitor
        self.baseline = baseline

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        acc = logs.get(self.monitor)
        if acc is not None:
            if acc >= self.baseline:
                print('Epoch %d: Reached baseline, terminating training' % (epoch))
                self.model.stop_training = True


class AnnEcg200():
    def __init__(self, train_db_file, test_db_file):
        self.train_df = None
        self.test_df = None
        self.train_db_file = train_db_file
        self.test_db_file = test_db_file
        self.im_train_df = None
        self.im_test_df = None
        self.norm_object = None
        self.st_object = None
        self.stn_object = None
        self.nst_object = None

    def create_dataframes(self):
        with open(self.train_db_file) as f:
            self.test_df = a2p.load(f)
            # print(self.test_df)

        with open(self.train_db_file) as f:
            self.train_df = a2p.load(f)
            # print(self.train_df)

    def prepare_normalized_train_data(self):
        self.norm_object = NormalizeFeatures(self.train_df, self.train_df.columns)
        self.norm_object.normalize_train_dataframe()
        normalized_train_df = self.norm_object.get_normalized_train_df()
        train_y = normalized_train_df[normalized_train_df.columns[-1]].values
        train_x = normalized_train_df.drop(labels='target@{-1,1}', axis=1).values
        # print(self.normalized_train_df.iloc[0])
        # print(train_x[0])
        # print(train_y)
        return train_x, train_y

    def prepare_standardized_train_data(self):
        features_df = self.train_df.drop(labels='target@{-1,1}', axis=1)
        self.st_object = StandardizeFeatures(features_df, features_df.columns)
        self.st_object.standardize_train_dataframe()
        standardized_train_df = self.st_object.get_standardized_train_df()
        train_y = self.train_df[self.train_df.columns[-1]].values
        train_x = standardized_train_df.values
        return train_x, train_y

    def prepare_stn_train_data(self):
        features_df = self.train_df.drop(labels='target@{-1,1}', axis=1)
        self.stn_object = StNFeatures(features_df, features_df.columns)
        self.stn_object.stn_train_dataframe()
        stn_train_df = self.stn_object.get_stn_train_df()
        train_y = self.train_df[self.train_df.columns[-1]].values
        train_x = stn_train_df.values
        return train_x, train_y

    def prepare_nst_train_data(self):
        features_df = self.train_df.drop(labels='target@{-1,1}', axis=1)
        self.nst_object = NStFeatures(features_df, features_df.columns)
        self.nst_object.nst_train_dataframe()
        nst_train_df = self.nst_object.get_nst_train_df()
        train_y = self.train_df[self.train_df.columns[-1]].values
        train_x = nst_train_df.values
        return train_x, train_y

    @staticmethod
    def train_model(train_x, train_y, epochs_to_train):
        overfitCallback = TerminateOnBaseline(monitor='acc', baseline=1)
        train_samle_length = len(train_x[0])
        model_fft_input = Input(shape=(train_samle_length,))
        model_fft_dense_1 = Dense(96, activation='relu')(model_fft_input)
        model_fft_dense_2 = Dense(48, activation='relu')(model_fft_dense_1)
        predict_out = Dense((1), activation='hard_sigmoid')(model_fft_dense_2)
        model_fft = Model(inputs=model_fft_input, outputs=predict_out)
        model_fft.compile(optimizer='rmsprop',
                          loss='binary_crossentropy',
                          metrics=['accuracy'])
        history = model_fft.fit(train_x, train_y, epochs=epochs_to_train, callbacks=[overfitCallback])
        return history, model_fft

    def get_normalized_model(self):
        x, y = self.prepare_normalized_train_data()
        _, model = self.train_model(x, y, 1000)
        return model

    def get_standardized_model(self):
        x, y = self.prepare_standardized_train_data()
        _, model = self.train_model(x, y, 1000)
        return model

    def get_stn_model(self):
        x, y = self.prepare_stn_train_data()
        _, model = self.train_model(x, y, 1000)
        return model

    def get_nst_model(self):
        x, y = self.prepare_nst_train_data()
        _, model = self.train_model(x, y, 1000)
        return model

    def test_normalized_model(self, model):
        normalized_test_df = self.norm_object.normalize_test_dataframe(self.test_df)
        # print(normalized_test_df)
        test_y = normalized_test_df[normalized_test_df.columns[-1]].values
        test_x = normalized_test_df.drop(labels='target@{-1,1}', axis=1).values
        scores = model.evaluate(x=test_x, y=test_y, verbose=1)
        return scores

    '''
    def test_standardized_model(self, model):
        standardized_test_df = self.st_object.standardize_test_dataframe(self.test_df)
        # print(standardized_test_df)
        test_y = standardized_test_df[standardized_test_df.columns[-1]].values
        test_x = standardized_test_df.drop(labels='target@{-1,1}', axis=1).values
        scores = model.evaluate(x=test_x, y=test_y, verbose=1)
        print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
    '''

    def test_standardized_model(self, model):
        test_fetures_df = self.test_df.drop(labels='target@{-1,1}', axis=1)
        standardized_test_df = self.st_object.standardize_test_dataframe(test_fetures_df)
        # print(standardized_test_df)
        test_y = self.test_df[self.test_df.columns[-1]].values
        test_x = standardized_test_df.values
        scores = model.evaluate(x=test_x, y=test_y, verbose=1)
        return scores

    def test_stn_model(self, model):
        test_fetures_df = self.test_df.drop(labels='target@{-1,1}', axis=1)
        stn_test_df = self.stn_object.stn_test_dataframe(test_fetures_df)
        # print(standardized_test_df)
        test_y = self.test_df[self.test_df.columns[-1]].values
        test_x = stn_test_df.values
        scores = model.evaluate(x=test_x, y=test_y, verbose=1)
        return scores

    def test_nst_model(self, model):
        test_fetures_df = self.test_df.drop(labels='target@{-1,1}', axis=1)
        nst_test_df = self.nst_object.nst_test_dataframe(test_fetures_df)
        # print(standardized_test_df)
        test_y = self.test_df[self.test_df.columns[-1]].values
        test_x = nst_test_df.values
        scores = model.evaluate(x=test_x, y=test_y, verbose=1)
        return scores

    def show_features_distribution(self, feature):
        fig, axs = plt.subplots(1, 5, sharey=True, tight_layout=True)
        axs[0].hist(pd.to_numeric(self.train_df[feature]).values, bins='auto')
        axs[1].hist(self.norm_object.get_normalized_train_df()[feature].values, bins='auto')
        axs[2].hist(self.st_object.get_standardized_train_df()[feature].values, bins='auto')
        axs[3].hist(self.stn_object.get_stn_train_df()[feature].values, bins='auto')
        axs[4].hist(self.nst_object.get_nst_train_df()[feature].values, bins='auto')
        print(pd.to_numeric(self.train_df[feature]).values)
        print(self.norm_object.get_normalized_train_df()[feature].values)
        print(self.st_object.get_standardized_train_df()[feature].values)
        print(self.stn_object.get_stn_train_df()[feature].values)
        print(self.nst_object.get_nst_train_df()[feature].values)
        plt.title("Histograms with 'auto' bins")
        plt.show()


if __name__ == "__main__":
    train_filepath = PurePath(os.getcwd(), 'ECG200', 'ECG200_TRAIN.arff')
    test_filepath = PurePath(os.getcwd(), 'ECG200', 'ECG200_TEST.arff')
    test_db_folder_name = PurePath(os.getcwd(), 'ecg200_images', 'test')
    ann_inst = AnnEcg200(train_filepath, test_filepath)
    ann_inst.create_dataframes()
    normalized_model = ann_inst.get_normalized_model()
    norm_train_df = ann_inst.norm_object.get_normalized_train_df()
    normalized_scores = ann_inst.test_normalized_model(normalized_model)
    standardized_model = ann_inst.get_standardized_model()
    standardized_scores = ann_inst.test_standardized_model(standardized_model)
    stn_model = ann_inst.get_stn_model()
    stn_scores = ann_inst.test_stn_model(stn_model)
    nst_model = ann_inst.get_nst_model()
    nst_scores = ann_inst.test_nst_model(nst_model)
    ann_inst.show_features_distribution(norm_train_df.columns[0])
    print("%s: %.2f%%" % (normalized_model.metrics_names[1], normalized_scores[1] * 100))
    print("%s: %.2f%%" % (standardized_model.metrics_names[1], standardized_scores[1] * 100))
    print("%s: %.2f%%" % (stn_model.metrics_names[1], stn_scores[1] * 100))
    print("%s: %.2f%%" % (nst_model.metrics_names[1], nst_scores[1] * 100))
    # print(ann_inst.st_object.get_standardized_train_df())
    # print(ann_inst.st_object.standardize_test_dataframe(ann_inst.test_df.drop(labels='target@{-1,1}', axis=1)))
