import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras import layers, optimizers
from tensorflow.keras.utils import plot_model
from tensorflow.keras.datasets.mnist import load_data
import argparse

#init the arguments
parser = argparse.ArgumentParser()
parser.add_argument('-b', '--batch', default=256, help='Batch Size for training(default: 256)')
parser.add_argument('-e', '--epoch', default=300, help='Number of epochs to run the training(default: 300)')
parser.add_argument('-a', '--alpha', default=0.0002, help='Learning Rate for AdamOptimizer(default: 0.0002)')
parser.add_argument('-b1', '--beta1', default=0.5, help='Beta1 for the AdamOptimizer(default: 0.5)')
parser.add_argument('-ne', '--n_epoch', default=60, help='Number of epochs after which to save model and get sample output(default: 60)')
args = vars(parser.parse_args())

#################################################################

np.random.seed(43)
#################################################################

#required Functions
#defining the generator model
def make_generator():
    model = tf.keras.Sequential()
    model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Reshape((7,7,256)))

    model.add(layers.Conv2DTranspose(128, (5,5), strides=1, padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5,5), strides=2 , padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(1, (5,5), strides = 2, padding='same', use_bias=False, activation='tanh'))

    return model

def make_discriminator(shape_input=[28,28,1]):
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5,5), strides=2, padding='same', input_shape=shape_input ))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5,5), strides=2, padding='same', input_shape=[28,28,1] ))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer=OPT, metrics=['accuracy'])

    return model

def img_output(noise, epoch=None):
    generated_img = generator.predict(noise)
    plt.figure(figsize=(8,8))
    for i,image in enumerate(generated_img):
        plt.subplot(10,10, i+1)
        plt.imshow(image.reshape((28,28)), cmap='gray_r')
        plt.axis('off')
    plt.show()
    if epoch!=None:
        plt.savefig(f'gan_image_{epoch}.png')

def make_gan(gmodel, dmodel):
    dmodel.trainable = False
    model = tf.keras.Sequential()
    model.add(gmodel)
    model.add(dmodel)
    model.compile(OPT, loss='binary_crossentropy')

    return model
#################################################################

#load dataset
(Xtrain,_), (_,_) = load_data()
Xtrain = np.array(Xtrain, dtype ='float32')
#visualize the results
plt.figure(figsize=(10,10))
for i in range(100):
    plt.subplot(10,10,i+1)
    plt.axis('off')
    plt.imshow(Xtrain[i+100],cmap='gray_r')
plt.show()
#preprocessing
Xtrain = Xtrain/255.0
Xtrain = Xtrain.reshape(-1, 28, 28, 1)
print(Xtrain.shape)
#################################################################

#intialise parameters
LATENT_DIM = 100 #noise vector for input in generator
BATCH_SIZE=args['batch']
STEPS_PER_EPOCH = int(Xtrain.shape[0]/BATCH_SIZE)
print("Steps per epoch====>>>> ",STEPS_PER_EPOCH)
EPOCHS= args['epoch']
OPT = optimizers.Adam(args['alpha'], args['beta1'])
#################################################################

generator = make_generator()
print(generator.summary())

discriminator = make_discriminator()
print(discriminator.summary())

gan = make_gan(generator, discriminator)
print(gan.summary())
#################################################################

#training the model
for epoch in range(EPOCHS):
    for batch in range(STEPS_PER_EPOCH):
        input_noise = np.random.normal(0,1, size=(BATCH_SIZE, LATENT_DIM))
        Xfake = generator.predict(input_noise)
        Xreal = Xtrain[np.random.randint(0, Xtrain.shape[0], size=BATCH_SIZE)]
        X = np.concatenate((Xreal, Xfake))
        y = np.zeros(2*BATCH_SIZE)
        y[:BATCH_SIZE] = 0.9 #label smoothing
        discriminator_loss = discriminator.train_on_batch(X, y)
        yGenerator = np.ones(BATCH_SIZE)
        generator_loss = gan.train_on_batch(input_noise, yGenerator)
    print(f'Epoch: {epoch}\tDiscriminator Loss: {discriminator_loss}\tGenreator Loss: {generator_loss}')
    if epoch%args['n_epoch']==0 and epoch!=0:
        img_output(np.random.normal(0, 1, size=(100, LATENT_DIM)))
        generator.save('gen{}.h5'.format(epoch))
        discriminator.save('dis{}.h5'.format(epoch))
