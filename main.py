import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import matplotlib.pyplot as plt

from functools import partial
from tqdm import tqdm
import os
from random import shuffle
from glob import glob

IMG_SIZE = (224, 224)  # Size of images

# Load image pixels from path with custom size
def load_image(path, target_size=IMG_SIZE):
    img = cv2.imread(path)[...,::-1]
    img = cv2.resize(img, target_size)
    return img #vgg16.preprocess_input(img)  # Preprocess for vgg16

# Training generator for inputs images
def fit_generator(files, batch_size=32):
    batch_size = min(batch_size, len(files))
    while True:
        shuffle(files)
        for k in range(len(files) // batch_size):
            i = k * batch_size
            j = i + batch_size
            if j > len(files):
                j = - j % len(files)
            x = np.array([load_image(path) for path in files[i:j]])
            y = np.array([1. if os.path.basename(path).startswith('dog') else 0.
                          for path in files[i:j]])
            yield (x, y)

# Testing generator for inputs images
def predict_generator(files):
    for path in tqdm(files):
        yield np.array([load_image(path)])
            
train_imgs = glob('data/train/*.jpg')
test_imgs = glob('data/test/*.jpg')

data_augmentation = keras.Sequential(
    [
     layers.RandomFlip("horizontal"),
     layers.RandomRotation(0.1),
     layers.RandomZoom(0.2)
     ]
)

shuffle(train_imgs)
batch_size = 48
train_generator = partial(fit_generator, files=train_imgs, batch_size=batch_size)

train_dataset = tf.data.Dataset.from_generator(
                    train_generator,
                    output_signature=(
                             tf.TensorSpec(shape=(batch_size, 224,224,3), dtype=tf.uint8),
                             tf.TensorSpec(shape=(batch_size,), dtype=tf.float64)
                             )
                    )

val_samples = 5  # число изображений в валидационной выборке
validation_data = next(fit_generator(train_imgs[:val_samples], val_samples))

plt.figure(figsize=(10,10))
for images, _ in train_dataset.take(1):
    for i in range(9):
        aug_imgs = data_augmentation(images)
        ax = plt.subplot(3,3,i+1)
        plt.imshow(aug_imgs[0].numpy().astype("uint8"))
        plt.axis("off")

#conv_base = keras.applications.vgg16.VGG16(
#    weights="imagenet",
#    include_top=False)
#conv_base.trainable = False
#conv_base.summary()
       
inputs = keras.Input(shape=(224,224,3))
x = data_augmentation(inputs)
#x = keras.applications.vgg16.preprocess_input(x)
#x = conv_base(x)
#x = layers.Flatten()(x)
#x = layers.Dense(256)(x)
x = layers.Rescaling(1./255)(x)
x = layers.Conv2D(filters=32, kernel_size=3, activation="relu")(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Conv2D(filters=64, kernel_size=3, activation="relu")(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Conv2D(filters=128, kernel_size=3, activation="relu")(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Conv2D(filters=256, kernel_size=3, activation="relu")(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Conv2D(filters=256, kernel_size=3, activation="relu")(x)
x = layers.Flatten()(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)

model = keras.Model(inputs=inputs, outputs=outputs)
model.compile(loss="binary_crossentropy",
              optimizer="rmsprop",
              metrics=["accuracy"])

model.summary()


callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath="vgg16.keras",
        save_best_only=True,
        monitor="val_loss"
        )]


# запускаем процесс обучения
history = model.fit(train_dataset,
              steps_per_epoch=25,  # число вызовов генератора за эпоху
              epochs=100,  # число эпох обучения
              validation_data=validation_data,
              callbacks=callbacks)


test_pred = model.predict(predict_generator(test_imgs))
    
import re

with open('submit.csv', 'w') as dst:
    dst.write('id,label\n')
    for path, score in zip(test_imgs, test_pred):
        dst.write('%s,%f\n' % (re.search('(\d+).jpg$', path).group(1), score))