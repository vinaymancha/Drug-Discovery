def DNN(input_shape=(128)):

    from keras import models
    from keras.layers import Dense
    from keras.layers.advanced_activations import LeakyReLU
    from keras.layers.normalization import BatchNormalization
    from keras.layers import Dropout
    from keras.layers.noise import GaussianNoise
    from keras.regularizers import l2

    
    model = models.Sequential()

    model.add(Dense(1000, activation='relu', input_shape=input_shape, kernel_regularizer=l2(0.0001)))
    model.add(Dropout(0.25))

    model.add(Dense(1000, activation='relu', kernel_regularizer=l2(0.0001)))
    model.add(Dropout(0.25))

    model.add(Dense(500, activation='relu', kernel_regularizer=l2(0.0001)))
    model.add(Dropout(0.10))

    model.add(Dense(1, activation=None, use_bias=True, kernel_regularizer=l2(0.0001)))

    # model.summary()

    return model
