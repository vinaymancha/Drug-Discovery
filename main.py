#Angry_Coders


from nutsflow import *
from nutsml import *
import matplotlib.pyplot as plt
from model import DNN
import numpy as np
import pandas as pd
from keras.optimizers import Adam, sgd
import sys


# Global variables
BATCH_SIZE = 64
EPOCH = 200
VAL_FREQ = 5
NET_ARCH = 'DNN'

data_root = 'C:/Users/vinay/Desktop/Drug Discovery/preprocessed/'

dataset_names = ['CB1', 'DPP4', 'HIVINT', 'HIVPROT', 'METAB', 'NK1', 'OX1', 'PGP', 'PPB', 'RAT_F',
                 'TDI', 'THROMBIN', 'OX2', '3A4', 'LOGD']

dataset_stats = pd.read_csv(data_root + 'dataset_stats.csv', header=None, names=['mean', 'std'], index_col=0)


def Rsqured_np(x, y):
    
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    avx = np.mean(x)
    avy = np.mean(y)

    num = np.sum((x - avx) * (y - avy))
    num = num * num

    denom = np.sum((x - avx) * (x - avx)) * np.sum((y - avy) * (y - avy))

    return num / denom


def RMSE_np(x, y):
    
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    n = x.shape[0]

    return np.sqrt(np.sum(np.square(x - y)) / n)
  
if __name__ == "__main__":
        for dataset_name in dataset_names:
        test_stat_hold = list()
        best_RMSE = float("inf")

        print 'Training on Data-set: ' + dataset_name
        test_file = data_root + dataset_name + '_test_disguised.csv'
        train_file = data_root + dataset_name + '_training_disguised.csv'

        data_train = ReadPandas(train_file, dropnan=True)
        Act_inx = data_train.dataframe.columns.get_loc('Act')
        feature_dim = data_train.dataframe.shape[1] - (Act_inx+1)

        # split randomly train and val
        data_train, data_val = data_train >> SplitRandom(ratio=0.8) >> Collect()
        data_test = ReadPandas(test_file, dropnan=True)

        def organize_features(sample):
            """
            reorganize the flow as a feature vector predictor pair
            :param sample: A row of data comming through the pipe
            :return: a tupe consising feature vector and predictor
            """
            y = [sample[Act_inx], ]
            features = list(sample[Act_inx+1:])
            return (features, y)

        build_batch = (BuildBatch(BATCH_SIZE)
                       .by(0, 'vector', float)
                       .by(1, 'number', float))

        if NET_ARCH == 'deep_net':
            model = deep_net(input_shape=(feature_dim,))
            opti = Adam(lr=0.0001, beta_1=0.5)
        elif NET_ARCH == 'merck_net':
            model = merck_net(input_shape=(feature_dim,))
            opti = sgd(lr=0.05, momentum=0.9, clipnorm=1.0)
        else:
            sys.exit("Network not defined correctly, check NET_ARCH. ")

        model.compile(optimizer=opti, loss='mean_squared_error', metrics=[Rsqured])

        def train_network_batch(sample):
            tloss = model.train_on_batch(sample[0], sample[1])
            return (tloss[0], tloss[1])

        def test_network_batch(sample):
            tloss = model.test_on_batch(sample[0], sample[1])
            return (tloss[0],)

        def predict_network_batch(sample):
            return model.predict(sample[0])

        scale_activators = lambda x: (
            x[0] * dataset_stats.loc[dataset_name, 'std'] + dataset_stats.loc[dataset_name, 'mean'])

        trues = data_val >> GetCols(Act_inx) >> Map(scale_activators) >> Collect()
        for e in range(1, EPOCH + 1):
            # training the network
            data_train >> Shuffle(1000) >> Map(organize_features) >> NOP(PrintColType()) >> build_batch >> Map(
                train_network_batch) >> NOP(Print()) >> Consume()

            # test the network every VAL_FREQ iteration
            if int(e) % VAL_FREQ == 0:
                preds = data_val >> Map(organize_features) >> build_batch >> Map(
                    predict_network_batch) >> Flatten() >> Map(scale_activators) >> Collect()

                RMSE_e = RMSE_np(preds, trues)
                Rsquared_e = Rsqured_np(preds, trues)
                print 'Dataset ' + dataset_name + ' Epoch ' + str(e), ' : RMSE = ' + str(
                    RMSE_e) + ', R-Squared = ' + str(Rsquared_e)
                test_stat_hold.append(('Epoch ' + str(e), RMSE_e, Rsquared_e))

                if RMSE_e < best_RMSE:
                    model.save_weights('./outputs/weights_' + dataset_name + '.h5')
                    best_RMSE = RMSE_e

        print "Calculating errors for test set ..."
        model.load_weights('./outputs/weights_' + dataset_name + '.h5')
        trues = data_test >> GetCols(Act_inx) >> Map(scale_activators) >> Collect()

        preds = data_test >> Map(organize_features) >> build_batch >> Map(predict_network_batch) >> Flatten() >> Map(
            scale_activators) >> Collect()

        RMSE_e = RMSE_np(preds, trues)
        Rsquared_e = Rsqured_np(preds, trues)
        print 'Dataset ' + dataset_name + ' Test : RMSE = ' + str(RMSE_e) + ', R-Squared = ' + str(Rsquared_e)
        test_stat_hold.append(('Final', RMSE_e, Rsquared_e))

        writer = WriteCSV('./outputs/test_errors_' + dataset_name + '.csv')
        test_stat_hold >> writer
  
  
  
