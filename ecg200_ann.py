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


class NormalizeFeatures():
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
        self._df = dataframe
        self._features = features
        self._normalized_df = None
        self._norms_df = None

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


class StandardizeFeatures():
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
        self._df = dataframe
        self._features = features
        self._standardized_df = None
        self._mean_std_df = None

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
        # print(self.normalized_train_df.iloc[0])
        # print(train_x[0])
        # print(train_y)
        return train_x, train_y

    '''
    def prepare_standardized_train_data(self):
        self.st_object = StandardizeFeatures(self.train_df, self.train_df.columns)
        self.st_object.standardize_train_dataframe()
        standardized_train_df = self.st_object.get_standardized_train_df()
        train_y = standardized_train_df[standardized_train_df.columns[-1]].values
        train_x = standardized_train_df.drop(labels='target@{-1,1}', axis=1).values
        # print(self.normalized_train_df.iloc[0])
        # print(train_x[0])
        # print(train_y)
        return train_x, train_y
    '''

    @staticmethod
    def train_model(train_x, train_y, epochs_to_train):
        overfitCallback = TerminateOnBaseline(monitor='acc', baseline=1)
        train_samle_length = len(train_x[0])
        model_fft_input = Input(shape=(train_samle_length,))
        model_fft_dense_1 = Dense(96, activation='relu')(model_fft_input)
        model_fft_dense_2 = Dense(48, activation='relu')(model_fft_dense_1)
        predict_out = Dense((1), activation='sigmoid')(model_fft_dense_2)
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


if __name__ == "__main__":
    train_filepath = PurePath(os.getcwd(), 'ECG200', 'ECG200_TRAIN.arff')
    test_filepath = PurePath(os.getcwd(), 'ECG200', 'ECG200_TEST.arff')
    test_db_folder_name = PurePath(os.getcwd(), 'ecg200_images', 'test')
    ann_inst = AnnEcg200(train_filepath, test_filepath)
    ann_inst.create_dataframes()
    normalized_model = ann_inst.get_normalized_model()
    normalized_scores = ann_inst.test_normalized_model(normalized_model)
    standardized_model = ann_inst.get_standardized_model()
    standardized_scores = ann_inst.test_standardized_model(standardized_model)
    print("%s: %.2f%%" % (normalized_model.metrics_names[1], normalized_scores[1] * 100))
    print("%s: %.2f%%" % (standardized_model.metrics_names[1], standardized_scores[1] * 100))
    # print(ann_inst.st_object.get_standardized_train_df())
    # print(ann_inst.st_object.standardize_test_dataframe(ann_inst.test_df.drop(labels='target@{-1,1}', axis=1)))
