import os
from pathlib import PurePath
import matplotlib.pyplot as plt
from pandas import DataFrame, Series
from keras.layers import Dense, Input, Dropout
from keras.models import Model
from keras.callbacks import Callback
from arff2pandas import a2p
from sklearn import preprocessing
import pandas as pd


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
        self.train_df_x = None
        self.train_df_y = None
        self.test_df_x = None
        self.test_df_y = None
        self.train_df_xs = None
        self.train_df_ys = None
        # self.test_df_xs = DataFrame()
        # self.test_df_ys = DataFrame()ne
        self.xscale_object = None
        self.yscale_object = None

        with open(test_db_file) as f:
            self.test_df = a2p.load(f)
            self.test_df_x = self.test_df.drop(labels='target@{-1,1}', axis=1)
            self.test_df_y = self.test_df[self.test_df.columns[-1]]
            # print(self.test_df)

        with open(train_db_file) as f:
            self.train_df = a2p.load(f)
            self.train_df_x = self.train_df.drop(labels='target@{-1,1}', axis=1)
            self.train_df_y = self.train_df[self.train_df.columns[-1]]
            # print(self.train_df)

    def normalize_dataframe_rows(self, df):
        row_normalized_dataframe = DataFrame()
        for row, series in df.iterrows():
            # print(series)
            series = pd.to_numeric(series)
            series = series.subtract(series.min())
            series = series.divide(series.max() - series.min())
            # print(new_series)
            row_normalized_dataframe = row_normalized_dataframe.append(series, ignore_index=False)
        row_normalized_dataframe = row_normalized_dataframe.reindex(df.columns, axis='columns')
        return row_normalized_dataframe

    def prepare_normalized_data(self):
        self.xscale_object = preprocessing.MinMaxScaler(feature_range=(0, 1), copy=True)
        self.yscale_object = preprocessing.MinMaxScaler(feature_range=(0, 1), copy=True)
        # scale x data
        self.train_df_xs = self.train_df_x
        self.train_df_xs[self.train_df_x.columns] = self.xscale_object.fit_transform(self.train_df_x[self.train_df_x.columns])
        self.test_df_xs = self.test_df_x
        self.test_df_xs[self.test_df_x.columns] = self.xscale_object.transform(self.test_df_x[self.test_df_x.columns])
        # scale y data
        self.train_df_ys = Series(self.yscale_object.fit_transform(self.train_df_y.values.reshape(-1, 1))[:, 0])
        self.test_df_ys = Series(self.yscale_object.transform(self.test_df_y.values.reshape(-1, 1))[:, 0])
        return self.train_df_xs.values.astype(dtype='float64'), self.train_df_ys.values.astype(dtype='float64')

    def prepare_standardized_data(self):
        self.xscale_object = preprocessing.StandardScaler(copy=True, with_mean=True, with_std=True)
        self.yscale_object = preprocessing.MinMaxScaler(feature_range=(0, 1), copy=True)
        # scale x data
        self.train_df_xs = self.train_df_x
        self.train_df_xs[self.train_df_x.columns] = self.xscale_object.fit_transform(self.train_df_x[self.train_df_x.columns])
        self.test_df_xs = self.test_df_x
        self.test_df_xs[self.test_df_x.columns] = self.xscale_object.transform(self.test_df_x[self.test_df_x.columns])
        # scale y data
        self.train_df_ys = Series(self.yscale_object.fit_transform(self.train_df_y.values.reshape(-1, 1))[:, 0])
        self.test_df_ys = Series(self.yscale_object.transform(self.test_df_y.values.reshape(-1, 1))[:, 0])

    @staticmethod
    def train_model(train_x, train_y, epochs_to_train):
        overfitCallback = TerminateOnBaseline(monitor='acc', baseline=1)
        train_samle_length = len(train_x[0])
        model_input = Input(shape=(train_samle_length,))
        model_dense_1 = Dense(96, activation='relu')(model_input)
        # model_dropout_1 = Dropout(rate=0.5)(model_dense_1)
        model_dense_2 = Dense(48, activation='relu')(model_dense_1)
        predict_out = Dense((1), activation='hard_sigmoid')(model_dense_2)
        model = Model(inputs=model_input, outputs=predict_out)
        model.compile(optimizer='rmsprop',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        history = model.fit(train_x, train_y, epochs=epochs_to_train, callbacks=[overfitCallback])
        return history, model

    def get_model(self):
        x = self.train_df_xs.values.astype(dtype='float64')
        y = self.train_df_ys.values.astype(dtype='float64')
        _, model = self.train_model(x, y, 1000)
        return model

    def test_model(self, model):
        # print(normalized_test_df)
        test_y = self.test_df_ys.values
        test_x = self.test_df_xs.values
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
    ann_inst = AnnEcg200(train_filepath, test_filepath)
    # ann_inst.prepare_normalized_data()
    ann_inst.prepare_standardized_data()
    model = ann_inst.get_model()
    scores = ann_inst.test_model(model)
    # print(ann_inst.norm_object.get_normalized_train_df())
    # print(ann_inst.norm_object.normalize_test_dataframe(ann_inst.test_df))
    print(ann_inst.train_df_xs)
    print(ann_inst.test_df_xs)
    # print(ann_inst.train_df_x['att1@NUMERIC'].std())
    print(ann_inst.train_df_ys)
    print(ann_inst.train_df_x['att1@NUMERIC'].max())
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
