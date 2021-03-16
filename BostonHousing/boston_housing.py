from tensorflow.keras.datasets import boston_housing
from tensorflow.keras.models import load_model
import numpy as np
import os
from utils.functions import build_sequential_model, is_model_exist


def train():
    (x_train, y_train), (x_test, y_test) = boston_housing.load_data()
    mean = x_train.mean(axis=0)
    std = x_train.mean(axis=0)
    x_train = x_train - mean
    x_test = x_test - mean
    x_train = x_train / std
    x_test = x_test / std
    new_model = build_sequential_model([x_train.shape[1], 128, 1],
                                       'adam', 'mse', ['mae'], None)
    print(new_model.summary())
    new_model.fit(x_train, y_train, epochs=100, validation_split=0.05, verbose=1)
    mse, mae = new_model.evaluate(x_test, y_test, verbose=0)
    print(f'MSE on test: {mse}, MAE: {mae}')
    new_model.save('boston_housing.h5')
    return new_model


if __name__ == '__main__':
    np.random.seed(42)
    model = load_model('boston_housing.h5') \
        if is_model_exist(os.getcwd(), 'boston_housing.h5') \
        else train()
