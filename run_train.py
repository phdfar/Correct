
import random
import path
from tensorflow import keras
import model
import accuracy
from keras.models import load_model
from keras.callbacks import CSVLogger
import os
import tensorflow as tf
global argss
import keras.backend as K
import matplotlib.pyplot as plt


import tensorflow.keras.backend as K


def dice_loss_sparse(y_true, y_pred):
    y_true = tf.squeeze(tf.cast(y_true, tf.int32), axis=-1)

    numerator = 2 * tf.reduce_sum(tf.one_hot(y_true, 4) * y_pred, axis=(1, 2))
    denominator = tf.reduce_sum(tf.one_hot(y_true, 4) + y_pred, axis=(1, 2))

    return (1 - numerator / denominator)
  
def tversky_loss_sparse(beta):
    def loss(y_true, y_pred):
        smooth = 1.
        y_true_pos = K.argmax(y_true, axis=-1)
        y_pred_pos = K.argmax(y_pred, axis=-1)
        true_pos = K.cast(K.equal(y_true_pos, y_pred_pos), 'float32')
        false_neg = K.sum((1 - true_pos) * y_true, axis=-1)
        false_pos = K.sum((1 - true_pos) * y_pred, axis=-1)
        tversky_index = (true_pos + smooth) / (true_pos + beta*false_neg + (1-beta)*false_pos + smooth)
        return 1.0 - K.mean(tversky_index)

    return loss
  
  
def dice_loss(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return 1. - score

def tversky_loss(beta):
    def loss(y_true, y_pred):
        smooth = 1.
        y_true_pos = K.flatten(y_true)
        y_pred_pos = K.flatten(y_pred)
        true_pos = K.sum(y_true_pos * y_pred_pos)
        false_neg = K.sum(y_true_pos * (1 - y_pred_pos))
        false_pos = K.sum((1 - y_true_pos) * y_pred_pos)
        return (1.0 - ((true_pos + smooth) / (true_pos + beta * false_neg + (1 - beta) * false_pos + smooth)))
    return loss
  
def start(args):

  allframe_train,allframe_val,allframe_test = path.getinfo_train(args)
  random.Random(1337).shuffle(allframe_train)
    
  dispatcher_loader={1:path.dataloader_2i,2:path.dataloader_2i}

  # Instantiate data Sequences for each split
  train_gen = dispatcher_loader[args.branch_input](args,allframe_train)
  val_gen = dispatcher_loader[args.branch_input](args,allframe_val)

  keras.backend.clear_session()
  if args.mode=='train':
    mymodel=model.network(args)
    mymodel.summary()

   
    if args.loss=='BCE':
      mymodel.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
    elif args.loss=='TVR':
      mymodel.compile(optimizer='adam', loss=tversky_loss_sparse(beta=0.5))
    elif args.loss=='DICE':
      mymodel.compile(optimizer='adam', loss=dice_loss_sparse)

  
    callbacks = [
        keras.callbacks.ModelCheckpoint(args.model_dir, save_best_only=True),CSVLogger(args.model_dir+'_log.csv', append=True, separator=',')
    ]
    if args.restore==True:
      mymodel = load_model(args.model_dir)
      
    mymodel.fit(train_gen, epochs=args.epoch, validation_data=val_gen, callbacks=callbacks)
  
  if args.mode=='test':
    test_gen = dispatcher_loader[args.branch_input](args,allframe_test)    
    mymodel = load_model(args.model_dir)
    mymodel.evaluate(test_gen);
    accuracy.start(mymodel,allframe_test,args.model_dir,args)

    
    

  """
  tap=[];vap=[];tep=[]
  
  for pathx in allframe_train:
    frameindex= list(pathx.keys())[0]
    imagepath = pathx[frameindex][0]
    tap.append(imagepath)
    
  for pathx in allframe_val:
    frameindex= list(pathx.keys())[0]
    imagepath = pathx[frameindex][0]
    vap.append(imagepath)
    
  for pathx in allframe_test:
    frameindex= list(pathx.keys())[0]
    imagepath = pathx[frameindex][0]
    tep.append(imagepath)
  
  import pickle
  with open('allpath2.pickle', 'wb') as handle:
    pickle.dump([tap,vap,tep], handle, protocol=pickle.HIGHEST_PROTOCOL)
    

  X, y = next(iter(test_gen))
  print(X.shape, y.shape)

  import pickle
  with open('loader.pickle', 'wb') as handle:
    pickle.dump([X,y], handle, protocol=pickle.HIGHEST_PROTOCOL)
  """

