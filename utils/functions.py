from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
import os


def build_sequential_model(layers, opt, loss, metrics, last_activator):
    if len(layers) < 2:
        raise ValueError('Should be at least 2 layers including inout')
    m = Sequential()
    for i in range(len(layers) - 1):
        inputs = layers[i]
        outputs = layers[i + 1]
        activation = 'relu' if i < len(layers) - 2 else last_activator
        if activation:
            m.add(Dense(outputs, activation=activation, input_dim=inputs))
        else:
            m.add(Dense(outputs, input_dim=inputs))
    m.compile(opt, loss=loss, metrics=metrics)
    return m


def is_model_exist(cwd, model_name):
    full_path = os.path.join(cwd, model_name)
    return os.path.exists(full_path)
