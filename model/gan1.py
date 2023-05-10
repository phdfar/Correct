from keras.models import Model
from keras.layers import Input, Concatenate, Conv2D, LeakyReLU, BatchNormalization, UpSampling2D, Activation, Lambda
from keras.optimizers import Adam
import tensorflow as tf

import random
from tensorflow import keras
import model
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
    
    # Input layers
    input_weak_mask = Input(shape=(384, 640, 1))
    input_rgb = Input(shape=(384, 640, 3))
    
    inputs = Concatenate()([input_weak_mask, input_rgb])
    
    # Encoder
    x = Conv2D(64, (5, 5), strides=(2, 2), padding='same')(inputs)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization()(x)
    
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
    
    # Decoder
    x = Conv2D(512, (5, 5), strides=(2, 2), padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization()(x)
    
    x = Conv2D(256, (5, 5), strides=(2, 2), padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization()(x)
    
    x = Conv2D(128, (5, 5), strides=(2, 2), padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization()(x)
    
    x = Conv2D(64, (5, 5), strides=(2, 2), padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization()(x)
    
    x = UpSampling2D(size=(2, 2))(x)
    x = Concatenate()([x, input_rgb])
    
    x = Conv2D(32, (5, 5), padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization()(x)
    
    output = Conv2D(1, (5, 5), activation='sigmoid', padding='same')(x)
    
    # Define the generator model
    generator = Model(inputs=[input_weak_mask, input_rgb], outputs=output, name='generator')
    
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
    output = Conv2D(1, (5, 5), activation='sigmoid', padding='same')(x)
    
    # Define the discriminator model
    discriminator = Model(inputs=input_gt_mask, outputs=output, name='discriminator')
    
    return discriminator

def build_gan(generator, discriminator):
  
  discriminator.trainable = False
  input_weak_mask = Input(shape=(384, 640, 1))
  input_rgb = Input(shape=(384, 640, 3))
  
  generated_masks = generator([input_weak_mask, input_rgb])
  
  validity = discriminator(generated_masks)
  
  combined = Model(inputs=[input_weak_mask, input_rgb], outputs=[generated_masks, validity], name='combined')
  combined.compile(loss=['mae', 'binary_crossentropy'], loss_weights=[100, 1], optimizer=Adam(lr=0.0002, beta_1=0.5))
  
  return combined


def model():

  generator = build_generator()
  discriminator = build_discriminator()
  gan = build_gan(generator, discriminator)
  
  print('Generator Summary:')
  generator.summary()
  
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
    
  
  steps_per_epoch = np.ceil(num_samples / args.batchsize)
  
  for epoch in range(args.epoch):
      print('Epoch {} of {}'.format(epoch+1, args.epoch))
  
      # Train discriminator
      for i in range(steps_per_epoch):
          # Get batch of real samples
          x_real, y_real = next(train_gen)
  
          # Train discriminator on real samples
          valid = np.ones((args.batchsize,) + discriminator.output_shape[1:])
          d_loss_real = discriminator.train_on_batch(y_real, valid)
  
          # Generate fake samples and train discriminator on them
          x_fake = [x_real[0], x_real[1]]
          y_fake = generator.predict(x_fake)
          fake = np.zeros((args.batchsize,) + discriminator.output_shape[1:])
          d_loss_fake = discriminator.train_on_batch(y_fake, fake)
  
          # Average loss
          d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
  
      # Train generator
      for i in range(steps_per_epoch):
          # Get batch of real samples
          x_real, y_real = next(train_gen)
  
          # Train generator using GAN model
          gan_loss = gan.train_on_batch(x_real, [y_real, np.ones((args.batchsize, 1))])
  
      # Print losses
      print('Discriminator Loss:', d_loss)
      print('Generator Loss:', gan_loss)

