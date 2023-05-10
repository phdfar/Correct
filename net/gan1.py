from keras.models import Model
from keras.layers import Input, Conv2DTranspose,Concatenate, Conv2D, LeakyReLU, BatchNormalization, UpSampling2D, Activation, Lambda
from keras.optimizers import Adam
import tensorflow as tf
from tensorflow.keras import layers

import random
from tensorflow import keras
from keras.models import load_model
from keras.callbacks import CSVLogger
import os
import tensorflow as tf
global argss
import keras.backend as K
import numpy as np

import tensorflow.keras.backend as K

def build_generator():
    """Builds the generator network"""
    
    
    input_rgb = keras.Input(shape=(384, 640, 3))
    input_weak_mask = keras.Input(shape=(384, 640, 1))

    ### [First half of the network: downsampling inputs] ###

    # Entry block
    x = layers.Conv2D(32, 3, strides=2, padding="same")(input_rgb)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    
    x = layers.Conv2D(32, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    
    
    y = layers.Conv2D(32, 3, strides=2, padding="same")(input_weak_mask)
    y = layers.BatchNormalization()(y)
    y = layers.Activation("relu")(y)
    
    y = layers.Conv2D(32, 3, strides=2, padding="same")(y)
    y = layers.BatchNormalization()(y)
    y = layers.Activation("relu")(y)

    z = layers.Concatenate()([x, y])
    
    previous_block_activation = z  # Set aside residual
    x = z

    for filters in [128, 64]:
        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.UpSampling2D(2)(x)

        # Project residual
        residual = layers.UpSampling2D(2)(previous_block_activation)
        residual = layers.Conv2D(filters, 1, padding="same")(residual)
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = layers.Conv2D(32, 3, padding="same",activation="relu")(x)
    x = layers.Conv2D(16, 3, padding="same",activation="relu")(x)
    x = layers.Conv2D(8, 3, padding="same",activation="relu")(x)
    x = layers.Conv2D(4, 3, padding="same",activation="relu")(x)

  
    output = layers.Conv2D(1, 3, activation="softmax", padding="same")(x)

    #output = Conv2D(1, (5, 5), activation='sigmoid', padding='same')(x)

    # Define the generator model
    generator = Model(inputs=[input_rgb,input_weak_mask], outputs=output, name='generator')

    return generator
    
def build_discriminator():
    """Builds the discriminator network"""
    
    # Input layer
    input_gt_mask = Input(shape=(384, 640, 1))
    
    # Convolutional layers
    x = Conv2D(64, (5, 5), strides=(2, 2), padding='same')(input_gt_mask)
    x = LeakyReLU(alpha=0.2)(x)
    
    x = Conv2D(128, (5, 5), strides=(2, 2), padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization()(x)
    
    x = Conv2D(256, (5, 5), strides=(2, 2), padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization()(x)
    
    x = Conv2D(512, (5, 5), strides=(2, 2), padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization()(x)
    
    x = Conv2D(1024, (5, 5), strides=(2, 2), padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization()(x)
    
    # Output layer
    x = Conv2D(1, (5, 5), activation='relu', padding='same')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(32)(x)
    x = layers.Dense(8)(x)
    output = layers.Dense(1,activation='sigmoid')(x)

    
    # Define the discriminator model
    discriminator = Model(inputs=input_gt_mask, outputs=output, name='discriminator')
    discriminator.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5))

    return discriminator

def build_gan(generator, discriminator):
  
  discriminator.trainable = False
  input_weak_mask = Input(shape=(384, 640, 1))
  input_rgb = Input(shape=(384, 640, 3))
  
  generated_masks = generator([input_rgb,input_weak_mask])
  
  validity = discriminator(generated_masks)
  
  combined = Model(inputs=[input_rgb,input_weak_mask], outputs=[generated_masks, validity], name='combined')
  combined.compile(loss=['mae', 'binary_crossentropy'], loss_weights=[100, 1], optimizer=Adam(lr=0.0002, beta_1=0.5))
  
  return combined


def model():

  generator = build_generator()
  
  print('Generator Summary:')
  generator.summary()
  
  discriminator = build_discriminator()
  gan = build_gan(generator, discriminator)

  print('\nDiscriminator Summary:')
  discriminator.summary()
  
  print('\nGAN Summary:')
  gan.summary()

  return generator,discriminator,gan

def run(args,train_gen,val_gen,num_samples):
  
  generator,discriminator,gan = model()
  
  callbacks = [
      keras.callbacks.ModelCheckpoint(args.model_dir, save_best_only=True),CSVLogger(args.model_dir+'_log.csv', append=True, separator=',')
  ]
  #if args.restore==True:
  #  mymodel = load_model(args.model_dir)
    
  
  steps_per_epoch = int(np.ceil(num_samples / args.batchsize))
  
  for epoch in range(args.epoch):
      print('Epoch {} of {}'.format(epoch+1, args.epoch))
  

      #print('train_gen',train_gen)
      # Train discriminator
      from tqdm import tqdm


      #train_iterable = iter(train_gen)

      pbar = tqdm(train_gen)
      
      for ins in pbar:
          # Get batch of real samples
          x_real, y_real = ins
  
          # Train discriminator on real samples
          valid = np.ones((args.batchsize,) + discriminator.output_shape[1:])
          d_loss_real = discriminator.train_on_batch(y_real, valid)
  
          # Generate fake samples and train discriminator on them
          x_fake = [x_real[0], x_real[1]]
          y_fake = generator.predict(x_fake,verbose=False)
          fake = np.zeros((args.batchsize,) + discriminator.output_shape[1:])
          d_loss_fake = discriminator.train_on_batch(y_fake, fake)
  
          # Average loss
          d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

          gan_loss = gan.train_on_batch(x_real, [y_real, np.ones((args.batchsize, 1))])

          #print('Discriminator Loss:', d_loss)
          #print('Generator Loss:', gan_loss)
          msg = str(gan_loss[0])+'--'+str(gan_loss[1])+'--'+str(gan_loss[2])
          pbar.set_description("Generator Loss " + msg )

  generator.save(args.model_dir)
  
"""
# Train generator
for ins in train_gen:
    # Get batch of real samples
    x_real, y_real = ins

    # Train generator using GAN model
    gan_loss = gan.train_on_batch(x_real, [y_real, np.ones((args.batchsize, 1))])
"""

      # Print losses
      

