from __future__ import print_function

import argparse
import os
import numpy as np
from PIL import Image
import math
from keras.models import Sequential
from keras.layers import Dense, Reshape
from keras.layers.core import Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D
from keras.optimizers import SGD
from keras.datasets import mnist

BATCH_SIZE = 128
NUM_EPOCHS = 100

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str)
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--num_epoch', type=int, default=NUM_EPOCHS)
    parser.add_argument('--pretty', dest='pretty', action='store_true')
    parser.set_defaults(pretty=False)
    args = parser.parse_args()
    return args

def generator_model():
    model = Sequential()
    model.add(Dense(input_dim=100, output_dim=1024))
    model.add(Activation('tanh'))
    model.add(Dense(128*7*7))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    model.add(Reshape((128,7,7), input_shape=(128*7*7,)))
    model.add(UpSampling2D(size=(2,2), dim_ordering="th"))
    model.add(Convolution2D(64,5,5, border_mode='same', dim_ordering="th"))
    model.add(Activation('tanh'))
    model.add(UpSampling2D(size=(2,2), dim_ordering="th"))
    model.add(Convolution2D(1,5,5, border_mode='same', dim_ordering="th"))
    model.add(Activation('tanh'))
    return model

def discriminator_model():
    model = Sequential()
    model.add(Convolution2D(64,5,5,
                            border_mode='same',
                            input_shape=(1,28,28),
                            dim_ordering="th"))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2,2), dim_ordering="th"))
    model.add(Convolution2D(128,5,5, border_mode='same', dim_ordering="th"))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2,2), dim_ordering="th"))
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('tanh'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    return model

def generator_containing_discriminator(generator, discriminator):
    model = Sequential()
    model.add(generator)
    discriminator.trainable = False
    model.add(discriminator)
    return model

def combine_images(generated_images):
    num = generated_images.shape[0]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num)/width))
    shape = generated_images.shape[2:]
    image = np.zeros((height*shape[0], width*shape[1]),
                      dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index/width)
        j = index % width
        image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1]] = \
            img[0, :, :]
    return image

def train(batch_size, num_epoch):
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = (X_train.astype(np.float32) - 127.5)/127.5
    X_train = X_train.reshape((X_train.shape[0], 1) + X_train.shape[1:])
    discriminator = discriminator_model()
    generator = generator_model()
    discriminator_on_generator = \
        generator_containing_discriminator(generator, discriminator)
    d_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
    g_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
    generator.compile(loss='binary_crossentropy', optimizer="SGD")
    discriminator_on_generator.compile(
        loss='binary_crossentropy', optimizer=g_optim)
    discriminator.trainable = True
    discriminator.compile(loss='binary_crossentropy', optimizer=d_optim)
    noise = np.zeros((batch_size, 100))
    for epoch in range(num_epoch):
        print("Epoch " + str(epoch+1) + "/" + str(num_epoch) +" :")
        print("Number of batches:", int(X_train.shape[0]/batch_size))
        for index in range(int(X_train.shape[0]/batch_size)):
            for i in range(batch_size):
                noise[i, :] = np.random.uniform(-1, 1, 100)
            image_batch = X_train[index*batch_size:(index+1)*batch_size]
            generated_images = generator.predict(noise, verbose=0)
            if index % 20 == 0:
                image = combine_images(generated_images)
                image = image*127.5+127.5
                Image.fromarray(image.astype(np.uint8)).save(
                    "images/"+str(epoch+1)+"_"+str(index+1)+".png")
            X = np.concatenate((image_batch, generated_images))
            y = [1] * batch_size + [0] * batch_size
            d_loss = discriminator.train_on_batch(X, y)
            print("Batch %d d_loss : %f" % (index+1, d_loss))
            for i in range(batch_size):
                noise[i, :] = np.random.uniform(-1, 1, 100)
            discriminator.trainable = False
            g_loss = discriminator_on_generator.train_on_batch(
                noise, [1] * batch_size)
            discriminator.trainable = True
            print("Batch %d g_loss : %f" % (index+1, g_loss))
            if index % 10 == 9:
                generator.save_weights('generator_weights', True)
                discriminator.save_weights('discriminator_weights', True)

def generate(batch_size, pretty=False):
    generator = generator_model()
    generator.compile(loss='binary_crossentropy', optimizer="SGD")
    generator.load_weights('generator_weights')
    if pretty:
        discriminator = discriminator_model()
        discriminator.compile(loss='binary_crossentropy', optimizer="SGD")
        discriminator.load_weights('discriminator_weights')
        noise = np.zeros((batch_size*20, 100))
        for i in range(batch_size*20):
            noise[i, :] = np.random.uniform(-1, 1, 100)
        generated_images = generator.predict(noise, verbose=1)
        d_pret = discriminator.predict(generated_images, verbose=1)
        index = np.arange(0, batch_size*20)
        index.resize((batch_size*20, 1))
        pre_with_index = list(np.append(d_pret, index, axis=1))
        pre_with_index.sort(key=lambda x: x[0], reverse=True)
        pretty_images = np.zeros((batch_size, 1) +
                               (generated_images.shape[2:]), dtype=np.float32)
        for i in range(int(batch_size)):
            idx = int(pre_with_index[i][1])
            pretty_images[i, 0, :, :] = generated_images[idx, 0, :, :]
        image = combine_images(pretty_images)
    else:
        noise = np.zeros((batch_size, 100))
        for i in range(batch_size):
            noise[i, :] = np.random.uniform(-1, 1, 100)
        generated_images = generator.predict(noise, verbose=1)
        image = combine_images(generated_images)
    image = image*127.5+127.5
    Image.fromarray(image.astype(np.uint8)).save(
        "images/generated_image.png")

def main():
    imagedir = "./images"
    if not os.path.exists(imagedir):
        os.makedirs(imagedir)

    args = get_args()
    if args.mode == 'train':
        train(batch_size = args.batch_size, num_epoch = args.num_epoch)
    elif args.mode == 'generate':
        generate(batch_size = args.batch_size, pretty = args.pretty)
    print("Done!")

if __name__ == "__main__":
    main()
