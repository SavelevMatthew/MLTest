from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras import utils
from PIL import Image
from io import BytesIO
from utils.colors import Colors
from utils.functions import build_sequential_model, is_model_exist
import numpy as np
import os
import requests
import sys


def train():
    # Загрузка данных
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    # flatten картинок + нормализация
    x_train = x_train.reshape(60000, 784)
    x_train = x_train / 255
    x_test = x_test.reshape(-1, 784)
    x_test = x_test / 255

    # метки (числа) в one-hot вектор
    y_train = utils.to_categorical(y_train, 10)
    y_test = utils.to_categorical(y_test, 10)

    # Создание модели
    new_model = build_sequential_model([784, 800, 10], 'SGD',
                                       'categorical_crossentropy',
                                       ['accuracy'], 'softmax')
    # Вывод показателей модели
    print(new_model.summary())
    # Обучение модели
    new_model.fit(x_train, y_train, batch_size=200, epochs=100, verbose=1,
                  validation_split=0.2)
    # Тестирование модели
    scores = new_model.evaluate(x_test, y_test, verbose=1)
    print(f'Average accuracy on test set is: {round(scores[1] * 100, 4)}')
    new_model.save('fashion_mnist.h5')

    # predictions = model.predict(x_train)
    # rnd = np.random.randint(0, 60000)
    # print(f'Random testing on sample [{rnd}]. Result: '
    #       f'{np.argmax(predictions[rnd]) == np.argmax(y_train[rnd])}')

    return new_model


if __name__ == '__main__':
    model = load_model('fashion_mnist.h5') \
        if is_model_exist(os.getcwd, 'fashion_mnist.h5') \
        else train()

    classes = ['T-Shirt', 'Trousers', 'Sweater', 'Dress', 'Coat',
               'Shoes', 'Shirt', 'Sneakers', 'Bag', 'Boots']

    c = Colors

    while True:
        print('=' * 64)
        url = input(f'{c.OK_CYAN}Input test image url '
                    f'(type "exit" or "e" to exit): {c.END_C}')
        if url.lower() == 'e' or url.lower() == 'exit':
            sys.exit()
        response = requests.get(url)
        if response.status_code != 200:
            raise ValueError('Incorrect url were given '
                             'or data receiving process failed!')
        img = Image.open(BytesIO(response.content))
        img = img.resize((28, 28))
        img = img.convert('L')

        # Предобработка картинки
        x = image.img_to_array(img)
        x = x.reshape(1, 784)
        x = 255 - x
        x = x / 255

        prediction = model.predict(x)
        class_number = int(np.argmax(prediction))
        print(f"Net thinks that {c.OK_GREEN}{c.UNDERLINE}{c.BOLD}"
              f"{classes[class_number]}{c.END_C} is on image!")
