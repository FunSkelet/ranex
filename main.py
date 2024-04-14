import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from multiprocessing import Process
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import PyDataset
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from PIL import Image

# Загрузка и предобработка своего датасета изображений
datagen = ImageDataGenerator(rescale=1. / 255)
train_generator = datagen.flow_from_directory(
    'DATA',  # Укажите путь к папке с вашими изображениями
    target_size=(128, 128),
    batch_size=1,
    class_mode='input',
    color_mode='grayscale',
)

# Размерность кодированного представления
encoding_dim = 128

# Входной слой
input_img = Input(shape=(128, 128))
# Кодированный слой
encoded = Dense(encoding_dim, activation='relu')(input_img)

# Декодированный слой
decoded = Dense(128, activation='sigmoid')(encoded)

# Модель автокодировщика
autoencoder = Model(input_img, decoded)

# Компиляция модели
autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy')

autoencoder.fit(train_generator, steps_per_epoch=len(train_generator), epochs=10)


# Генерация представления своего изображения с помощью обученного кодировщика
def generate_representation(img_path):
    img = Image.open(img_path).convert('L')  # Предполагаем, что изображение в оттенках серого
    img = img.resize((128, 128))
    x = np.array(img)
    x = x.astype('float32') / 255.
    x = x.reshape(1, 128, 128)
    encoded_img = autoencoder.predict(x)
    return encoded_img


# Генерация похожих изображений из датасета с использованием обученного декодировщика
def generate_similar_images(encoded_img, num_similar=10):
    decoded_imgs = autoencoder.predict(encoded_img)
    return decoded_imgs[:num_similar]


# Загрузка своего изображения и получение его представления
your_img_path = 'DATA/Минералогия5399.jpg'  # Укажите путь к вашему изображению
your_encoded_img = generate_representation(your_img_path)

# Генерация и вывод похожих изображений
similar_images = generate_similar_images(your_encoded_img)
plt.figure(figsize=(20, 4))
for i in range(len(similar_images)):
    ax = plt.subplot(1, len(similar_images), i + 1)
    plt.imshow(similar_images[i].reshape(128, 128), cmap='gray')
    plt.title(f'Similar Image {i + 1}')
    plt.axis('off')
plt.show()
