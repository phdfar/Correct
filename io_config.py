from tensorflow.keras.preprocessing.image import load_img
import numpy as np
import cv2
#from models.Spectral import load

def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def run(myself,path):

  rgbpath = myself.baseinput+'train/JPEGImages/'+path['img']
  gtpath = myself.basepath+'/'+path['gt']
  weakpath = myself.basepath+'/'+path['weak']
  if myself.task=='diff':
    gtpath = myself.basepath+'/'+path['diff']
    
  img  = np.asarray(load_img(rgbpath, target_size=myself.img_size,grayscale=False))
  weak = cv2.imread(weakpath,0)
  gt = cv2.imread(gtpath,0)

  dim = (myself.img_size[1],myself.img_size[0])
  gt = cv2.resize(gt, dim, interpolation = cv2.INTER_NEAREST)
  weak = cv2.resize(weak, dim, interpolation = cv2.INTER_NEAREST)

  gt = np.expand_dims(gt,2)
  weak = np.expand_dims(weak,2)

  """
  frameindex= list(path.keys())[0]
  imagepath = path[frameindex][0]
  seq = path[frameindex][1]
  flagmulti = path[frameindex][2]
  
  
  img = np.asarray(load_img(myself.baseinput+'train/'+imagepath, target_size=myself.img_size,grayscale=False))
  
  
  sp = imagepath.split('/'); name=sp[-1].replace('.jpg','.pth.npy');eigpath = sp[-2]+'_'+name;
  eig = np.load(myself.baseinput2+eigpath) #data/VOC2012/eigs/laplacian/
    
  dim = (myself.img_size[1],myself.img_size[0])
  eig1 = cv2.resize(eig[:,:,1], dim, interpolation = cv2.INTER_NEAREST)
  eig1 = NormalizeData(eig1)
  
  eig1[eig1<=0.15]=0;eig1[eig1>0.15]=1;
  eig1 = np.expand_dims(eig1,2)

  x = [img,eig1]
  
  namey = eigpath.replace('.pth.npy','.png')

  y = myself.goodness_score[namey]
  y = np.expand_dims(y,0)
  """

  return [img,weak],gt
