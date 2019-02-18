# Copyright 2016 Vladimir Nekrasov
import numpy as np
import tensorflow as tf
from medpy.io import load
import random
from multiprocessing.dummy import Pool as ThreadPool
from skimage.transform import resize
def image_scaling(img, label):
  """Randomly scales the images between 0.5 to 1.5 times the original size.

  Args:
    img: A tensor of size [batch_size, height_in, width_in, channels]
    label: A tensor of size [batch_size, height_in, width_in]

  Returns:
    A tensor of size [batch_size, height_out, width_out, channels], and another
    tensor of size [batch_size, height_out, width_out]
  """
  scale = tf.random_uniform(
      [1], minval=0.5, maxval=2.0, dtype=tf.float32, seed=None)
  h_new = tf.to_int32(tf.to_float(tf.shape(img)[0]) * scale)
  w_new = tf.to_int32(tf.to_float(tf.shape(img)[1]) * scale)
  new_shape = tf.squeeze(tf.stack([h_new, w_new]), squeeze_dims=[1])
  img = tf.image.resize_images(img, new_shape)
  # Rescale labels by nearest neighbor sampling.
  label = tf.image.resize_nearest_neighbor(
      tf.expand_dims(label, 0), new_shape)
  label = tf.squeeze(label, squeeze_dims=[0])
   
  return img, label


def image_mirroring(img, label):
  """Randomly horizontally mirrors the images and their labels.

  Args:
    img: A tensor of size [batch_size, height_in, width_in, channels]
    label: A tensor of size [batch_size, height_in, width_in]

  Returns:
    A tensor of size [batch_size, height_in, width_in, channels], and another
    tensor of size [batch_size, height_in, width_in]
  """
  distort_left_right_random = tf.random_uniform(
      [1], 0, 1.0, dtype=tf.float32)
  distort_left_right_random = distort_left_right_random[0]

  mirror = tf.less(tf.stack([1.0, distort_left_right_random, 1.0]), 0.5)
  mirror = tf.boolean_mask([0, 1, 2], mirror)
  img = tf.reverse(img, mirror)
  label = tf.reverse(label, mirror)

  return img, label


def crop_and_pad_image_and_labels(image,
                                  label,
                                  crop_h,
                                  crop_w,
                                  ignore_label=255,
                                  random_crop=True):
  """Randomly crops and pads the images and their labels.

  Args:
    img: A tensor of size [batch_size, height_in, width_in, channels]
    label: A tensor of size [batch_size, height_in, width_in]
    crop_h: A number indicating the height of output data.
    crop_w: A number indicating the width of output data.
    ignore_label: A number indicating the indices of ignored label.
    random_crop: enable/disable random_crop for random cropping.

  Returns:
    A tensor of size [batch_size, height_out, width_out, channels], and another
    tensor of size [batch_size, height_out, width_out, 1]
  """
  # Needs to be subtracted and later added due to 0 padding.
  label = tf.cast(label, dtype=tf.float32)
  label = label - ignore_label 

  # Concatenate images with labels, which makes random cropping easier.
  combined = tf.concat(axis=2, values=[image, label]) 
  image_shape = tf.shape(image)
  combined_pad = tf.image.pad_to_bounding_box(
      combined,
      0,
      0,
      tf.maximum(crop_h, image_shape[0]),
      tf.maximum(crop_w, image_shape[1]))
    
  last_image_dim = tf.shape(image)[-1]
  last_label_dim = tf.shape(label)[-1]

  if random_crop:
    combined_crop = tf.random_crop(combined_pad, [crop_h,crop_w,4])
  else:
    combined_crop = tf.image.resize_image_with_crop_or_pad(
        combined_pad,
        crop_h,
        crop_w)

  img_crop = combined_crop[:, :, :last_image_dim]
  label_crop = combined_crop[:, :, last_image_dim:]
  label_crop = label_crop + ignore_label
  label_crop = tf.cast(label_crop, dtype=tf.uint8)
    
  # Set static shape so that tensorflow knows shape at running. 
  img_crop.set_shape((crop_h, crop_w, 3))
  label_crop.set_shape((crop_h,crop_w, 1))

  return img_crop, label_crop  


def read_labeled_image_list(data_dir, data_list):
  """Reads txt file containing paths to images and ground truth masks.
    
  Args:
    data_dir: A string indicating the path to the root directory of images
      and masks.
    data_list: A string indicating the path to the file with lines of the form
      '/path/to/image /path/to/label'.
       
  Returns:
    Two lists with all file names for images and masks, respectively.
  """
  f = open(data_list, 'r')
  images = []
  masks = []
  for line in f:
    try:
      image, mask = line.strip("\n").split(' ')
    except ValueError: # Adhoc for test.
      image = mask = line.strip("\n")
    images.append(data_dir + image)
    masks.append(data_dir + mask)
  return images, masks




def load_fast_files(data_dir):
    img_list = []
    tumor_list = []
    minindex_list = []
    maxindex_list = []
    tumorlines = []
    tumoridx = []
    liveridx = []
    liverlines = []
    for idx in range(131):
        img, img_header = load(data_dir+ '/myTrainingData/volume-' + str(idx) + '.nii')
        tumor, tumor_header = load(data_dir + '/TrainingData/segmentation-' + str(idx) + '.nii')
        img_list.append(img)
        tumor_list.append(tumor)

        maxmin = np.loadtxt(data_dir + '/myTrainingDataTxt/LiverBox/box_' + str(idx) + '.txt', delimiter=' ')
        minindex = maxmin[0:3]
        maxindex = maxmin[3:6]
        minindex = np.array(minindex, dtype='int')
        maxindex = np.array(maxindex, dtype='int')
        minindex[0] = max(minindex[0] - 3, 0)
        minindex[1] = max(minindex[1] - 3, 0)
        minindex[2] = max(minindex[2] - 3, 0)
        maxindex[0] = min(img.shape[0], maxindex[0] + 3)
        maxindex[1] = min(img.shape[1], maxindex[1] + 3)
        maxindex[2] = min(img.shape[2], maxindex[2] + 3)
        minindex_list.append(minindex)
        maxindex_list.append(maxindex)
        f1 = open(data_dir + '/myTrainingDataTxt/TumorPixels/tumor_' + str(idx) + '.txt', 'r')
        tumorline = f1.readlines()
        tumorlines.append(tumorline)
        tumoridx.append(len(tumorline))
        f1.close()
        f2 = open(data_dir + '/myTrainingDataTxt/LiverPixels/liver_' + str(idx) + '.txt', 'r')
        liverline = f2.readlines()
        liverlines.append(liverline)
        liveridx.append(len(liverline))
        f2.close()
    return img_list, tumor_list, tumorlines, liverlines, tumoridx, liveridx, minindex_list, maxindex_list

def load_test_files(data_dir):
    img_list = []
    tumor_list = []
    minindex_list = []
    maxindex_list = []
    tumorlines = []
    tumoridx = []
    liveridx = []
    liverlines = []
    for idx in range(70):
        img, img_header = load(data_dir+ '/myTestingData/volume-' + str(idx) + '.nii')
        tumor, tumor_header = load(data_dir + '/myTestingData/segmentation-' + str(idx) + '.nii')
        img_list.append(img)
        tumor_list.append(tumor)

        maxmin = np.loadtxt(data_dir + '/myTrainingDataTxt/LiverBox/box_' + str(idx) + '.txt', delimiter=' ')
        minindex = maxmin[0:3]
        maxindex = maxmin[3:6]
        minindex = np.array(minindex, dtype='int')
        maxindex = np.array(maxindex, dtype='int')
        minindex[0] = max(minindex[0] - 3, 0)
        minindex[1] = max(minindex[1] - 3, 0)
        minindex[2] = max(minindex[2] - 3, 0)
        maxindex[0] = min(img.shape[0], maxindex[0] + 3)
        maxindex[1] = min(img.shape[1], maxindex[1] + 3)
        maxindex[2] = min(img.shape[2], maxindex[2] + 3)
        minindex_list.append(minindex)
        maxindex_list.append(maxindex)
        f1 = open(data_dir + '/myTrainingDataTxt/TumorPixels/tumor_' + str(idx) + '.txt', 'r')
        tumorline = f1.readlines()
        tumorlines.append(tumorline)
        tumoridx.append(len(tumorline))
        f1.close()
        f2 = open(data_dir + '/myTrainingDataTxt/LiverPixels/liver_' + str(idx) + '.txt', 'r')
        liverline = f2.readlines()
        liverlines.append(liverline)
        liveridx.append(len(liverline))
        f2.close()
    return img_list, tumor_list, tumorlines, liverlines, tumoridx, liveridx, minindex_list, maxindex_list

class ImageReader(object):
  """
  Generic ImageReader which reads images and corresponding
  segmentation masks from the disk, and enqueues them into
  a TensorFlow queue.
  """

  def __init__(self, data_dir, data_list, input_size,
               random_scale, random_mirror, random_crop,
               ignore_label, img_mean,train=True):
    """
    Initialise an ImageReader.
          
    Args:
      data_dir: path to the directory with images and masks.
      data_list: path to the file with lines of the form
                 '/path/to/image /path/to/mask'.
      input_size: a tuple with (height, width) values, to which all the
                  images will be resized.
      random_scale: whether to randomly scale the images.
      random_mirror: whether to randomly mirror the images.
      ignore_label: index of label to ignore during the training.
      img_mean: vector of mean colour values.

    Returns:
      A tensor of size [batch_size, height_out, width_out, channels], and
      another tensor of size [batch_size, height_out, width_out]
    """
    self.data_dir = data_dir
    self.data_list = data_list
    self.input_size = input_size
    self.mean=img_mean
    self.train=train
    #self.image_list, self.label_list = self.get_all_lists(self.data_list)##
    #self.images = tf.convert_to_tensor(self.image_list, dtype=tf.string)
    #self.labels = tf.convert_to_tensor(self.label_list, dtype=tf.string)
    #self.queue = tf.train.slice_input_producer(
    #    [self.images, self.labels],
    #    shuffle=input_size is not None) # not shuffling if it is val
    #self.image, self.label = read_images_from_disk(
    #    self.queue,
    #    self.input_size,
    #    random_scale,
    #    random_mirror,
    #    random_crop,
    #    ignore_label,
    #    img_mean)

    self.image, self.label, self.tumorlines, self.liverlines, self.tumoridx, self.liveridx, \
        self.minindex_list, self.maxindex_list=load_fast_files(self.data_dir)




  def get_all_lists(self,list_idx):
      images = []
      masks = []
      for idx in list_idx:
          images.append(self.data_dir + '/myTrainingData/volume-'+str(idx)+'.nii')
          masks.append(self.data_dir + '/myTrainingData/segmentation-'+str(idx)+'.nii')
      return images, masks
  def length(self):
      l=0
      for mi,ma in zip(self.minindex_list,self.maxindex_list):
          l+=ma-mi
      return l

  def read_images_from_disk(self,input_queue,
                            input_size,
                            random_scale,
                            random_mirror,
                            random_crop,
                            ignore_label,
                            img_mean):
      """Reads one image and its corresponding label and perform pre-processing.

      Args:
        input_queue: A tensorflow queue with paths to the image and its mask.
        input_size: A tuple with entries of height and width. If None, return
          images of original size.
        random_scale: enable/disable random_scale for randomly scaling images
          and their labels.
        random_mirror: enable/disable random_mirror for randomly and horizontally
          flipping images and their labels.
        ignore_label: A number indicating the index of label to ignore.
        img_mean: A vector indicating the mean colour values of RGB channels.

      Returns:
        Two tensors: the decoded image and its mask.
      """
      liverlist = [32, 34, 38, 41, 47, 87, 89, 91, 105, 106, 114, 115, 119]

      count = input_queue[0]
      img = self.readimage(count)
      label = self.readlabel(count)
      img = tf.convert_to_tensor(img)
      # image_batch=(image_batch+350)/300
      label = tf.convert_to_tensor(label, dtype=tf.int32)

      minindex = self.readminindex_list(count)
      maxindex = self.readmaxindex_list[count]
      #num = np.random.randint(0, 6)
      num = tf.random_uniform(
          [1], minval=0, maxval=6, dtype=tf.int32, seed=None)
      if num[0] < 3 or (count.eval() in liverlist):
          lines = self.liverlines[count]
          numid = self.liveridx[count]
      else:
          lines = self.tumorlines[count]
          numid = self.tumoridx[count]
      Parameter_List=[img, label, lines, numid, minindex, maxindex]
      img,label=self.load_seq_crop_data_masktumor_try(Parameter_List)
      if input_size is not None:
          h, w = input_size

          # Randomly scale the images and labels.
          if random_scale:
              img, label = image_scaling(img, label)

          # Randomly mirror the images and labels.
          if random_mirror:
              img, label = image_mirroring(img, label)

          # Randomly crops the images and labels.
          img, label = crop_and_pad_image_and_labels(
              img, label, h, w, ignore_label, random_crop
          )

      return img, label
  def load_seq_crop_data_masktumor_try(self, Parameter_List):
      img = Parameter_List[0]
      tumor = Parameter_List[1]
      lines = Parameter_List[2]
      numid = Parameter_List[3]
      minindex = Parameter_List[4]
      maxindex = Parameter_List[5]
      #  randomly scale
      scale = np.random.uniform(0.8, 1.2)
      if not self.train:
          scale=1
      deps = int(self.input_size[0] * scale)
      rows = int(self.input_size[1] * scale)
      cols = 3

      sed = np.random.randint(1, numid)
      cen = lines[sed - 1]
      cen = np.fromstring(cen, dtype=int, sep=' ')

      a = min(max(minindex[0] + deps / 2, cen[0]), maxindex[0] - deps / 2 - 1)
      b = min(max(minindex[1] + rows / 2, cen[1]), maxindex[1] - rows / 2 - 1)
      c = min(max(minindex[2] + cols / 2, cen[2]), maxindex[2] - cols / 2 - 1)  # (a,b,c) is the real center
      cropp_img = img[a - deps / 2:a + deps / 2, b - rows / 2:b + rows / 2,
                  c - cols / 2: c + cols / 2 + 1].copy()
      cropp_tumor = tumor[a - deps / 2:a + deps / 2, b - rows / 2:b + rows / 2,
                    c - cols / 2:c + cols / 2 + 1].copy()

      # idx_map_x=np.arange(s[0], e[0], (e[0]-s[0])/(self.input_size[0]))
      # if idx_map_x.shape[0]>self.input_size[0]:
      #    idx_map_x=idx_map_x[:-1]
      # idx_map_x=np.repeat(np.reshape(idx_map_x,[1,self.input_size[0]]),self.input_size[0],0)
      # idx_map_y=np.arange(s[1], e[1], (e[1]-s[1])/(self.input_size[1]))
      # if idx_map_y.shape[0]>self.input_size[1]:
      #    idx_map_y=idx_map_y[:-1]
      # idx_map_y=np.repeat(np.reshape(idx_map_y,[self.input_size[1],1]),self.input_size[1],1)
      # idx_map=np.stack([idx_map_x,idx_map_y],-1)
      #cropp_img+=250.1
      cropp_img=(cropp_img+250)*255/500
      cropp_img-=self.mean
      # randomly flipping
      if self.train:
          flip_num = np.random.randint(0, 8)
          if flip_num == 1:
              cropp_img = np.flipud(cropp_img)
              cropp_tumor = np.flipud(cropp_tumor)
          elif flip_num == 2:
              cropp_img = np.fliplr(cropp_img)
              cropp_tumor = np.fliplr(cropp_tumor)
          elif flip_num == 3:
              cropp_img = np.rot90(cropp_img, k=1, axes=(1, 0))
              cropp_tumor = np.rot90(cropp_tumor, k=1, axes=(1, 0))
          elif flip_num == 4:
              cropp_img = np.rot90(cropp_img, k=3, axes=(1, 0))
              cropp_tumor = np.rot90(cropp_tumor, k=3, axes=(1, 0))
          elif flip_num == 5:
              cropp_img = np.fliplr(cropp_img)
              cropp_tumor = np.fliplr(cropp_tumor)
              cropp_img = np.rot90(cropp_img, k=1, axes=(1, 0))
              cropp_tumor = np.rot90(cropp_tumor, k=1, axes=(1, 0))
          elif flip_num == 6:
              cropp_img = np.fliplr(cropp_img)
              cropp_tumor = np.fliplr(cropp_tumor)
              cropp_img = np.rot90(cropp_img, k=3, axes=(1, 0))
              cropp_tumor = np.rot90(cropp_tumor, k=3, axes=(1, 0))
          elif flip_num == 7:
              cropp_img = np.flipud(cropp_img)
              cropp_tumor = np.flipud(cropp_tumor)
              cropp_img = np.fliplr(cropp_img)
              cropp_tumor = np.fliplr(cropp_tumor)
      # idx_map=resize(idx_map, (self.input_size[0],self.input_size[1],2), order=3, mode='constant', cval=0, clip=True, preserve_range=True)
      cropp_tumor = resize(cropp_tumor, (self.input_size[0], self.input_size[1], 3), order=0, mode='edge',
                           cval=0, clip=True, preserve_range=True)
      cropp_img = resize(cropp_img, (self.input_size[0], self.input_size[1], 3), order=3, mode='constant',
                         cval=0, clip=True, preserve_range=True)

      return cropp_img, cropp_tumor[:, :, 1]
  def dequeue(self, num_elements):
    """Packs images and labels into a batch.
        
    Args:
      num_elements: A number indicating the batch size.
          
    Returns:
      A tensor of size [batch_size, height_out, width_out, 3], and
      another tensor of size [batch_size, height_out, width_out, 1]
    """
    liverlist = [32, 34, 38, 41, 47, 87, 89, 91, 105, 106, 114, 115, 119]
    X = np.zeros((num_elements, self.input_size[0], self.input_size[1], 3), dtype='float32')
    Y = np.zeros((num_elements, self.input_size[0], self.input_size[1], 1), dtype='int16')
    Parameter_List = []
    thread_num=3
    for idx in range(num_elements):
        count = random.choice(self.data_list)

        img = self.image[count]
        tumor = self.label[count]
        minindex = self.minindex_list[count]
        maxindex = self.maxindex_list[count]
        num = np.random.randint(0, 5)
        if num < 3 or (count in liverlist):
            lines = self.liverlines[count]
            numid = self.liveridx[count]
        else:
            lines = self.tumorlines[count]
            numid = self.tumoridx[count]
        Parameter_List.append([img, tumor, lines, numid, minindex, maxindex])
    pool = ThreadPool(thread_num)
    result_list =pool.map(self.load_seq_crop_data_masktumor_try,Parameter_List)
    pool.close()
    pool.join()
    for idx in range(len(result_list)):
        X[idx, :, :, :] = result_list[idx][0]
        Y[idx, :, :, 0] = result_list[idx][1]
    return X, Y
