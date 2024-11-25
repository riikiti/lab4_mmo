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
                batch_size = 32))  # * РѕРїС‚РёРјРёР·РёСЂСѓРµРјС‹Р№ РїР°СЂР°РјРµС‚СЂ

model.add(Conv2D(
            filters = 120,
            kernel_size = (3,3),
            strides = (1,1), # РЁР°Рі РґРІРёР¶РµРЅРёСЏ СЏРґСЂР° СЃРІС‘СЂС‚РєРё
            padding = 'same', # Р”РѕР±Р°РІР»РµРЅРёРµ Рє РєР°СЂС‚РёРЅРєРµ РїРѕР»РµР№ same РёР»Рё valid (РєР°СЂС‚РёРЅРєР° СѓРјРµРЅСЊС€РёС‚СЃСЏ)
            activation = 'relu'))

# С‚СѓС‚ РјРѕР¶РЅРѕ РґРѕР±Р°РІРёС‚СЊ РµС‰С‘ РѕРґРёРЅ, РёР»Рё РЅРµ РѕРґРёРЅ СЃРІС‘СЂС‚РѕС‡РЅС‹Р№ СЃР»РѕР№

model.add(MaxPooling2D(# РџРѕРЅРёР¶РµРЅРёРµ СЂР°Р·РјРµСЂРЅРѕСЃС‚Рё РёР·РѕР±СЂР°Р¶РµРЅРёСЏ
            pool_size = (2,2),
            strides = (2,2)))

model.add(Flatten()) # РџСЂРµРѕР±СЂР°Р·РѕРІР°РЅРёРµ РґР°РЅРЅС‹С… РІ РѕРґРЅРѕРјРµСЂРЅСѓСЋ РїРѕСЃР»РµРґРѕРІР°С‚РµР»СЊРЅРѕСЃС‚СЊ.

model.add(Dense(
            units = 10,
            activation='softmax'))


#print(model.summary())

model.compile(optimizer='adam', # РђР»РіРѕСЂРёС‚Рј РіСЂР°РґРёРµРЅС‚РЅРѕРіРѕ СЃРїСѓСЃРєР° *
             loss='categorical_crossentropy', # РєСЂРѕСЃСЃСЌРЅС‚СЂРѕРїРёСЏ Р»СѓС‡С€Рµ РїРѕРґС…РѕРґРёС‚ РґР»СЏ Р·Р°РґР°С‡ РєР»Р°СЃСЃРёС„РёРєР°С†РёРё *
             metrics=['accuracy']) # РїРѕРєР°Р·С‹РІР°РµС‚ % РїСЂРёРјРµСЂРѕРІ, РєРѕС‚РѕСЂР°СЏ РїСЂР°РІРёР»СЊРЅРѕ РїСЂРѕРєР»Р°СЃСЃРёС„РёС†РёСЂРѕРІР°РЅР°


model.fit(x_train, y_train_cat,
                epochs = 2,
                verbose = 0)

score = model.evaluate(x_train, y_train_cat, verbose=0)

print("РћС€РёР±РєР° РЅР° РѕР±СѓС‡Р°СЋС‰РµР№ РІС‹Р±РѕСЂРєРµ: %f" % score[0])
print("РўРѕС‡РЅРѕСЃС‚СЊ РЅР° РѕР±СѓС‡Р°СЋС‰РµР№ РІС‹Р±РѕСЂРєРµ: %f" % score[1])

score = model.evaluate(x_test, y_test_cat, verbose=0)

print("РћС€РёР±РєР° РЅР° С‚РµСЃС‚РѕРІРѕР№ РІС‹Р±РѕСЂРєРµ: %f" % score[0])
print("РўРѕС‡РЅРѕСЃС‚СЊ РЅР° С‚РµСЃС‚РѕРІРѕР№ РІС‹Р±РѕСЂРєРµ: %f" % score[1])

# РЅР°СѓС‡РёС‚СЊСЃСЏ РІС‹РІРѕРґРёС‚СЊ РєР°СЂС‚РёРЅРєРё, РіРґРµ РќРЎ РѕС€РёР±Р»Р°СЃСЊ
# РїРѕРїС‹С‚Р°С‚СЃСЏ РѕР±С‹С‡РЅС‹Рј РїРµСЂС†РµРїС‚СЂРѕРЅРѕРј
# РјР°РєСЃРёРјР°Р»СЊРЅРѕ СѓР»СѓС‡С€РёС‚СЊ РїРѕРєР°Р·Р°С‚РµР»Рё
# РїРѕРєР°Р·С‹РІР°С‚СЊ СЂРµР·СѓР»СЊС‚Р°С‚ С‚РѕС‡РЅРѕСЃС‚Рё Рё РѕР±СЉСЏСЃРЅРёС‚СЊ, РїРѕС‡РµРјСѓ Рё С‡С‚Рѕ РїРѕРјРѕРіР»Рѕ СѓР»СѓС‡С€РёС‚СЊ.
# РјРѕР¶РЅРѕ РґРѕР±Р°РІР»СЏС‚СЊ РґСЂСѓРіРёРµ СЃРІС‘СЂС‚РѕС‡РЅС‹Рµ СЃР»РѕРё (РјРѕР¶РЅРѕ РґСЂСѓРі Р·Р° РґСЂСѓРіРѕРј), РјРѕРґРёС„РёС†РёСЂРѕРІР°С‚СЊ РїРµСЂС†РµРїС‚СЂРѕРЅРЅСѓСЋ С‡Р°СЃС‚СЊ.