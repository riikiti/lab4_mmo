import os
import numpy as np
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from tensorflow import keras
from keras.datasets import mnist
from keras import Sequential, Input
from keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D



(x_train, y_train), (x_test, y_test) = mnist.load_data('mnist.npz')

#print(x_train[0])
x_train = x_train / 255
x_test = x_test / 255


#print(y_train[0])
y_train_cat = keras.utils.to_categorical(y_train, 10)
y_test_cat = keras.utils.to_categorical(y_test, 10)

#print(y_train_cat[0])
model = Sequential()

model.add(Input(shape=(28, 28, 1),
                batch_size = 32))  # * оптимизируемый параметр

model.add(Conv2D(
            filters = 120,
            kernel_size = (3,3),
            strides = (1,1), # Шаг движения ядра свёртки
            padding = 'same', # Добавление к картинке полей same или valid (картинка уменьшится)
            activation = 'relu'))

model.add(Conv2D(
            filters = 120,
            kernel_size = (3,3),
            strides = (1,1), # Шаг движения ядра свёртки
            padding = 'same', # Добавление к картинке полей same или valid (картинка уменьшится)
            activation = 'relu'))

# тут можно добавить ещё один, или не один свёрточный слой

model.add(MaxPooling2D(# Понижение размерности изображения
            pool_size = (2,2),
            strides = (2,2)))

model.add(Flatten()) # Преобразование данных в одномерную последовательность.

model.add(Dense(
            units = 10,
            activation='softmax'))


#print(model.summary())

model.compile(optimizer='adam', # Алгоритм градиентного спуска *
             loss='categorical_crossentropy', # кроссэнтропия лучше подходит для задач классификации *
             metrics=['accuracy']) # показывает % примеров, которая правильно проклассифицирована


model.fit(x_train, y_train_cat,
                epochs = 1,
                verbose = 0)

score = model.evaluate(x_train, y_train_cat, verbose=0)

print("Ошибка на обучающей выборке: %f" % score[0])
print("Точность на обучающей выборке: %f" % score[1])

score = model.evaluate(x_test, y_test_cat, verbose=0)

print("Ошибка на тестовой выборке: %f" % score[0])
print("Точность на тестовой выборке: %f" % score[1])

# научиться выводить картинки, где НС ошиблась
# попытатся обычным перцептроном
# максимально улучшить показатели
# показывать результат точности и объяснить, почему и что помогло улучшить.
# можно добавлять другие свёрточные слои (можно друг за другом), модифицировать перцептронную часть.