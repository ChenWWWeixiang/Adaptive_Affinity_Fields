from __future__ import print_function
from scipy import ndimage
import argparse
import os
import time
import math
from medpy.io import load,save
import tensorflow as tf
import numpy as np
import scipy.io
import scipy.misc
from PIL import Image
from skimage import measure
from seg_models.models.pspnet import pspnet_resnet101 as model
from seg_models.image_reader import ImageReader
import utils.general

IMG_MEAN = np.array((122.675, 116.669, 104.008), dtype=np.float32)

os.environ['CUDA_VISIBLE_DEVICES']='0,1'
def get_arguments():
  """Parse all the arguments provided from the CLI.
    
  Returns:
    A list of parsed arguments.
  """
  parser = argparse.ArgumentParser(
      description='Inference for Semantic Segmentation')
  parser.add_argument('--data', type=str, default='/home/data1/U-net/H-DenseUNet/data/myTestingData/test-volume-',
                      help='/path/to/dataset.')
  parser.add_argument('--liver_path', type=str, default='/home/data1/U-net/H-DenseUNet/livermask/',
                      help='/path/to/dataset.')
  parser.add_argument('--data-list', type=str, default='',
                      help='/path/to/datalist/file.')
  parser.add_argument('--input-size', type=str, default='528,528',
                      help='Comma-separated string with H and W of image.')
  parser.add_argument('--strides', type=str, default='512,512',
                      help='Comma-separated string with strid   es of H and W.')
  parser.add_argument('--num-classes', type=int, default=3,
                      help='Number of classes to predict.')
  parser.add_argument('--ignore-label', type=int, default=255,
                      help='Index of label to ignore.')
  parser.add_argument('--restore-from', type=str, default='snapshot_v2/model.ckpt-30000',
                      help='Where restore model parameters from.')
  parser.add_argument('--save_path', type=str, default='outputs/',
                      help='/path/to/save/predictions.')
  parser.add_argument('--mean', type=int, default=48,
                      help='/path/to/colormap/file.')
  parser.add_argument('-thres_liver', type=float, default=0.5)
  parser.add_argument('-thres_tumor', type=float, default=0.9)
  return parser.parse_args()

def loadckpt(saver, sess, ckpt_path):
  """Load the trained weights.
  
  Args:
    saver: TensorFlow saver object.
    sess: TensorFlow session.
    ckpt_path: path to checkpoint file with parameters.
  """ 
  saver.restore(sess, ckpt_path)
  print('Restored model parameters from {}'.format(ckpt_path))
def parse_commastr(str_comma):
  """Read comma-sperated string.
  """
  if '' == str_comma:
    return None
  else:
    a, b =  map(int, str_comma.split(','))

  return [a,b]
def slice(x, h1, h2):
    """ Define a tensor slice function
    """
    return x[:, :, :, h1:h2,:]
def slice_last(x):

    x = x[:,:,:,:,0]
    return x
def slice2d(x, h1, h2):

    tmp = x[h1:h2,:,:,:]
    tmp = np.transpose(tmp, [1, 2, 0, 3])
    tmp = np.expand_dims(tmp, 0)
    return tmp

def tans2d(img_input,input_cols):
    input2d = slice(img_input,h1= 0, h2=2)
    single = slice(img_input,h1= 0, h2=1)
    input2d = np.concatenate([single, input2d], axis=3)
    for i in range(input_cols - 2):
        input2d_tmp = slice(img_input,h1= i, h2= i + 3)
        input2d = np.concatenate([input2d, input2d_tmp], axis=0)
        if i == input_cols - 3:
            final1 =slice(img_input,h1= input_cols-2, h2= input_cols)
            final2 = slice(img_input, h1=input_cols-1, h2=input_cols)
            final = np.concatenate([final1, final2], axis=3)
            input2d = np.concatenate([input2d, final], axis=0)
    input2d = slice_last(input2d)
    return input2d

def trans3d(classifer2d,input_cols):
    res2d = slice2d(classifer2d, 0,1)
    for j in range(input_cols - 1):
        score = slice2d(classifer2d, j+1,j+2)
        res2d = np.concatenate([res2d, score], axis=3)
    return res2d

def main():
  """Create the model and start the Inference process.
  """
  args = get_arguments()

  #TODO:5. postprocession and save
  # Parse image processing arguments.
  print('get model!')
  input_size = parse_commastr(args.input_size)
  strides = parse_commastr(args.strides)
  assert(input_size is not None and strides is not None)
  h, w = input_size
  #innet_size = (int(math.ceil(h/8)), int(math.ceil(w/8)))

  # Create input tensor to the Network.
  crop_image_batch = tf.placeholder(
      name='crop_image_batch',
      shape=[8,input_size[0],input_size[1],3],
      dtype=tf.float32)

  # Create network and output prediction.
  outputs = model(crop_image_batch,
                  args.num_classes,
                  False,
                  True)

  # Grab variable names which should be restored from checkpoints.
  restore_var = [
    v for v in tf.global_variables() if 'crop_image_batch' not in v.name]
    
  # Output predictions.
  output = outputs[-1]
  output = tf.image.resize_bilinear(
      output,
      tf.shape(crop_image_batch)[1:3,])
  output = tf.nn.softmax(output, dim=3)
    
  # Set up tf session and initialize variables. 
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  sess = tf.Session(config=config)
  init = tf.global_variables_initializer()
    
  sess.run(init)
  sess.run(tf.local_variables_initializer())
    
  # Load weights.
  loader = tf.train.Saver(var_list=restore_var)
  if args.restore_from is not None:
    loadckpt(loader, sess, args.restore_from)
    
  # Start queue threads.
  #threads = tf.train.start_queue_runners(coord=coord, sess=sess)

  for id in range(70):
      print('-' * 30)
      print('preprocessing test data...' + str(id))
      print('-' * 30)

      #1. load a medpy file from mem
      imgs_test, img_test_header = load(args.data + str(id) + '.nii')
      mm=np.zeros((input_size[0],input_size[1],imgs_test.shape[2]))
      mm[:imgs_test.shape[0],:imgs_test.shape[1],:imgs_test.shape[2]]=imgs_test
      imgs_test=mm


      #  load liver mask
      mask, mask_header = load(args.liver_path + str(id) + '-ori.nii')
      mask[mask == 2] = 1
      mask = ndimage.binary_dilation(mask, iterations=1).astype(mask.dtype)
      print('-' * 30)
      print('Predicting masks on test data...' + str(id))
      print('-' * 30)
      index = np.where(mask == 1)
      mini = np.min(index, axis=-1)
      maxi = np.max(index, axis=-1)

      batch = 1
      img_deps = input_size[0]
      img_rows = input_size[1]
      img_cols = 8

      window_cols = (img_cols / 4)
      count = 0
      box_test = np.zeros((batch, img_deps, img_rows, img_cols, 1), dtype="float32")
      x = imgs_test.shape[0]
      y = imgs_test.shape[1]
      z = imgs_test.shape[2]
      right_cols = int(min(z, maxi[2] + 10) - img_cols)
      left_cols = max(0, min(mini[2] - 5, right_cols))
      score = np.zeros((x, y, z, 3), dtype='float32')
      score_num = np.zeros((x, y, z, 3), dtype='int16')
      for cols in xrange(left_cols, right_cols + window_cols, window_cols):
          # print ('and', z-img_cols,z)
          if cols > z - img_cols:
              patch_test = imgs_test[0:img_deps, 0:img_rows, z - img_cols:z]
              box_test[count, :, :, :, 0] = patch_test
              incol = box_test.shape[3]
              box_testt = tans2d(box_test, incol)
              box_testt = (box_testt + 250) * 255 / 500
              box_testt -= np.array((122.675, 122.669, 122.008), dtype=np.float32)
              # print ('final', img_cols-window_cols, img_cols)
              feed_dict = {crop_image_batch: box_testt}
              patch_test_mask = sess.run(output, feed_dict=feed_dict)

              patch_test_mask = trans3d(patch_test_mask, incol)
              patch_test_mask = patch_test_mask[:, :, :, 1:-1, :]

              for i in xrange(batch):
                  score[0:img_deps, 0:img_rows, z - img_cols + 1:z - 1, :] += patch_test_mask[i]
                  score_num[0:img_deps, 0:img_rows, z - img_cols + 1:z - 1, :] += 1
          else:
              patch_test = imgs_test[0:img_deps, 0:img_rows, cols:cols + img_cols]
              # print(patch_test.shape)
              box_test[count, :, :, :, 0] = patch_test
              incol = box_test.shape[3]
              box_testt = tans2d(box_test, incol)
              box_testt = (box_testt + 250) * 255 / 500
              box_testt -= np.array((122.675, 122.669, 122.008), dtype=np.float32)
              feed_dict = {crop_image_batch: box_testt}
              patch_test_mask = sess.run(output, feed_dict=feed_dict)
              patch_test_mask = trans3d(patch_test_mask, incol)

              patch_test_mask = patch_test_mask[:, :, :, 1:-1, :]

              for i in xrange(batch):
                  score[0:img_deps, 0:img_rows, cols + 1:cols + img_cols - 1, :] += patch_test_mask[i]
                  score_num[0:img_deps, 0:img_rows, cols + 1:cols + img_cols - 1, :] += 1
      score = score / (score_num + 1e-4)
      result1 = score[:512, :512, :, 1]
      result2 = score[:512, :512, :, 2]

      result1[result1 >= args.thres_liver] = 1
      result1[result1 < args.thres_liver] = 0
      result2[result2 >= args.thres_tumor] = 1
      result2[result2 < args.thres_tumor] = 0
      result1[result2 == 1] = 1

      print('-' * 30)
      print('Postprocessing on mask ...' + str(id))
      print('-' * 30)

      #  preserve the largest liver
      Segmask = result2
      box = []
      [liver_res, num] = measure.label(result1, return_num=True)
      region = measure.regionprops(liver_res)
      for i in range(num):
          box.append(region[i].area)
      label_num = box.index(max(box)) + 1
      liver_res[liver_res != label_num] = 0
      liver_res[liver_res == label_num] = 1

      #  preserve the largest liver
      mask = ndimage.binary_dilation(mask, iterations=1).astype(mask.dtype)
      box = []
      [liver_labels, num] = measure.label(mask, return_num=True)
      region = measure.regionprops(liver_labels)
      for i in range(num):
          box.append(region[i].area)
      label_num = box.index(max(box)) + 1
      liver_labels[liver_labels != label_num] = 0
      liver_labels[liver_labels == label_num] = 1
      liver_labels = ndimage.binary_fill_holes(liver_labels).astype(int)

      #  preserve tumor within ' largest liver' only
      Segmask = Segmask * liver_labels
      Segmask = ndimage.binary_fill_holes(Segmask).astype(int)
      Segmask = np.array(Segmask, dtype='uint8')
      liver_res = np.array(liver_res, dtype='uint8')
      liver_res = ndimage.binary_fill_holes(liver_res).astype(int)
      liver_res[Segmask == 1] = 2
      liver_res = np.array(liver_res, dtype='uint8')
      save(liver_res, args.save_path + 'test-segmentation-' + str(id) + '.nii', img_test_header)
    
if __name__ == '__main__':
    main()
