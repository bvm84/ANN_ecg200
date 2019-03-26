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

    def normalize_dataframe(self):
        self._normalized_df = DataFrame()
        self._norms_df = DataFrame()
        for col in self._features:
            normalized_vector, vector_min, vector_max = self.normalize(self._df[col])
            self._normalized_df.insert(loc=len(self._normalized_df.columns), column=col, value=normalized_vector)
            self._norms_df.insert(loc=len(self._norms_df.columns), column=col, value=(vector_min, vector_max))
            self._norms_df.rename(index={0: 'feature_min', 1: 'feaure_max'}, inplace=True)

    def get_normalized_df(self):
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

    def standardize_dataframe(self):
        self._standardized_df = DataFrame()
        self._mean_std_df = DataFrame()
        for col in self._features:
            standardized_vector, vector_mean, vector_std = self.standardize(self._df[col])
            self._standardized_df.insert(loc=len(self._standardized_df.columns), column=col, value=standardized_vector)
            self._mean_std_df.insert(loc=len(self._mean_std_df.columns), column=col, value=(vector_mean, vector_std))
            self._mean_std_df.rename(index={0: 'feature_mean', 1: 'feaure_std'}, inplace=True)

    def get_standardized_df(self):
        if self._standardized_df.empty is False:
            return self._standardized_df
        else:
            raise Exception('Standardized dataframe is empty')

    def get_mean_std_df(self):
        if self._mean_std_df.empty is False:
            return self._mean_std_df
        else:
            raise Exception('Standardized dataframe is empty')


class TerminateOnBaseline(Callback):
    """Callback that terminates training when either acc or val_acc reaches a specified baseline
    """

    def __init__(self, monitor='acc', baseline=0.9):
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


class ConvPrECG():
    def __init__(self, train_db_folder, test_db_folder):
        self.train_df = None
        self.test_df = None
        self.train_db_folder = train_db_folder
        self.test_db_folder = test_db_folder
        self.im_train_df = None
        self.im_test_df = None
        self.normalized_train_df = None
        self.norms_df = None

    def create_dataframes(self):
        test_path = PurePath(os.getcwd(), 'ECG200', 'ECG200_TEST.arff')

        with open(test_path) as f:
            self.test_df = a2p.load(f)
            # print(self.test_df)

        train_path = PurePath(os.getcwd(), 'ECG200', 'ECG200_TRAIN.arff')
        with open(train_path) as f:
            self.train_df = a2p.load(f)
            # print(self.train_df)

    def normalize_dataframe(self):
        norm_inst = NormalizeFeatures(self.train_df, self.train_df.columns)
        norm_inst.normalize_dataframe()
        self.normalized_train_df = norm_inst.get_normalized_df()
        self.norms_df = norm_inst.get_norms_df()
        print(self.normalized_train_df)
        print(self.norms_df)


    @staticmethod
    def train_model(train_df, epochs_to_train):
        # overfitCallback = EarlyStopping(monitor='acc', min_delta=0.00001, mode='min', baseline=0.99, patience=3)
        overfitCallback = TerminateOnBaseline(monitor='acc', baseline=0.99)
        train_samle_length = len(tran_df['x'][0])
        model_fft_input = Input(shape=(train_samle_length,))
        model_fft_dense_1 = Dense(72, activation='relu')(model_fft_input)
        model_fft_dense_2 = Dense(36, activation='relu')(model_fft_dense_1)
        predict_out = Dense(len(labels_list), activation='softmax')(model_fft_dense_2)
        model_fft = Model(inputs=model_fft_input, outputs=predict_out)
        model_fft.compile(optimizer=tf.train.AdamOptimizer(),
                          loss='sparse_categorical_crossentropy',
                          metrics=['accuracy'])
        history = model_fft.fit(x_train, y_train, epochs=epochs_to_train, callbacks=[overfitCallback])
        # print(history.history)
        return history, model_fft

    def get_model(self):
        x = np.asarray(self.im_train_df['x'].tolist())
        y = np.asarray(self.im_train_df['y'].tolist())
        model = self.train_2model(x, y, 50)
        return model

    def test_model(self, model):
        x = np.asarray(self.im_test_df['x'].tolist())
        y = np.asarray(self.im_test_df['y'].tolist())
        scores = model.evaluate(x, y, verbose=1)
        print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

class TerminateOnBaseline(Callback):
    """Callback that terminates training when either acc or val_acc reaches a specified baseline
    """

    def __init__(self, monitor='acc', baseline=0.9):
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

if __name__ == "__main__":
    train_db_folder_name = PurePath(os.getcwd(), 'ecg200_images', 'train')
    test_db_folder_name = PurePath(os.getcwd(), 'ecg200_images', 'test')
    inst = ConvPrECG(train_db_folder_name, test_db_folder_name)
    inst.create_dataframes()
    inst.normalize_train_dataframe()
    '''
    data = {'weigth': [1, 2, 0, 0, 0, 0, 1],
            'Age': [20, 21, 19, 18, 1, 2, 102]}
    test_dataframe = DataFrame(data)
    norm_inst = NormalizeFeatures(test_dataframe, test_dataframe.columns)
    norm_inst.normalize_dataframe()
    print(norm_inst.get_normalized_df())
    print(norm_inst.get_norms_df())
    stand_inst = StandardizeFeatures(test_dataframe, test_dataframe.columns)
    stand_inst.standardize_dataframe()
    print(stand_inst.get_standardized_df())
    print(stand_inst.get_mean_std_df())
    '''
