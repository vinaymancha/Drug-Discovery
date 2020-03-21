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
  
  
  
