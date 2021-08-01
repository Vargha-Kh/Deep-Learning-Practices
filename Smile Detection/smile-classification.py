from __future__ import print_function
# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os
import sys
import tarfile
from scipy import ndimage
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle
import random
import hashlib

num_classes = 2
train_folders = ['datasets/train_folder/0', 'datasets/train_folder/1']
test_folders = ['datasets/test_folder/0', 'datasets/test_folder/1']

image_size = 64
pixel_depth = 255.0
image_depth = 3

def load_image(folder, min_num_images):
  """Load the image for a single smile/non-smile lable."""
  image_files = os.listdir(folder)
  dataset = np.ndarray(shape=(len(image_files), image_size, image_size, image_depth),
                         dtype=np.float32)
  image_index = 0
  for image in os.listdir(folder):
    image_file = os.path.join(folder, image)
    try:
      image_data = (ndimage.imread(image_file).astype(float) - 
                    pixel_depth / 2) / pixel_depth
      # print(image_data)
      if image_data.shape != (image_size, image_size, image_depth):
        raise Exception('Unexpected image shape: %s' % str(image_data.shape))
      dataset[image_index, :, :, :] = image_data
      image_index += 1
    except IOError as e:
      print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')
    
  num_images = image_index
  dataset = dataset[0:num_images, :, :]
  if num_images < min_num_images:
    raise Exception('Many fewer images than expected: %d < %d' %
                    (num_images, min_num_images))
    
  print('Full dataset tensor:', dataset.shape)
  print('Mean:', np.mean(dataset))
  print('Standard deviation:', np.std(dataset))
  return dataset

#################################################################          

def maybe_pickle(data_folders, min_num_images_per_class, force=False):
  dataset_names = []
  for folder in data_folders:
    set_filename = folder + '.pickle'
    dataset_names.append(set_filename)
    if os.path.exists(set_filename) and not force:
      # You may override by setting force=True.
      print('%s already present - Skipping pickling.' % set_filename)
    else:
      print('Pickling %s.' % set_filename)
      dataset = load_image(folder, min_num_images_per_class)
      try:
        with open(set_filename, 'wb') as f:
          pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
      except Exception as e:
        print('Unable to save data to', set_filename, ':', e)
  
  return dataset_names

train_datasets = maybe_pickle(train_folders, 1200)
test_datasets = maybe_pickle(test_folders, 500)

#################################################################

# merge and prune data
def make_arrays(nb_rows, img_size, img_depth=3):
  if nb_rows:
    dataset = np.ndarray((nb_rows, img_size, img_size, img_depth), dtype=np.float32)
    labels = np.ndarray(nb_rows, dtype=np.int32)
  else:
    dataset, labels = None, None
  return dataset, labels

def merge_datasets(pickle_files, train_size, valid_size=0):
  num_classes = len(pickle_files)

  valid_dataset, valid_labels = make_arrays(valid_size, image_size)
  train_dataset, train_labels = make_arrays(train_size, image_size)
  vsize_per_class = valid_size // num_classes # 400
  tsize_per_class = train_size // num_classes # 200
    
  start_v, start_t = 0, 0
  end_v, end_t = vsize_per_class, tsize_per_class
  end_l = vsize_per_class+tsize_per_class
  for label, pickle_file in enumerate(pickle_files):
    # print(pickle_file)
    
    try:
      with open(pickle_file, 'rb') as f:
        smile_nonsmile_set = pickle.load(f)
        # print(smile_nonsmile_set.shape)

        # let's shuffle the smile / nonsmile class
        # to have random validation and training set
        np.random.shuffle(smile_nonsmile_set)
        if valid_dataset is not None:
          valid_smile_nonsmile = smile_nonsmile_set[:vsize_per_class, :, :]
          valid_dataset[start_v:end_v, :, :] = valid_smile_nonsmile
          valid_labels[start_v:end_v] = label
          start_v += vsize_per_class
          end_v += vsize_per_class
                    
        train_smile_nonsmile = smile_nonsmile_set[vsize_per_class:end_l, :, :]
        train_dataset[start_t:end_t, :, :] = train_smile_nonsmile
        train_labels[start_t:end_t] = label

        start_t += tsize_per_class
        end_t += tsize_per_class
    except Exception as e:
      print('Unable to process data from', pickle_file, ':', e)
      raise
  return valid_dataset, valid_labels, train_dataset, train_labels

# train_size = 2800
train_size = 2400
valid_size = 600
test_size = 600

_, _, train_dataset, train_labels = merge_datasets(
  train_datasets, train_size)
valid_dataset, valid_labels, test_dataset, test_labels = merge_datasets(
  test_datasets, test_size, valid_size)

# print('Training:', train_dataset.shape, train_labels.shape)
# print('Validation:', valid_dataset.shape, valid_labels.shape)
# print('Testing:', test_dataset.shape, test_labels.shape)

#################################################################

def randomize(dataset, labels):
  permutation = np.random.permutation(labels.shape[0])
  shuffled_dataset = dataset[permutation,:,:]
  shuffled_labels = labels[permutation]
  return shuffled_dataset, shuffled_labels
train_dataset, train_labels = randomize(train_dataset, train_labels)
test_dataset, test_labels = randomize(test_dataset, test_labels)
valid_dataset, valid_labels = randomize(valid_dataset, valid_labels)

# pretty_labels = {0: 'non-smile', 1: 'smile'}
# def disp_sample_dataset(dataset, labels):
#   print(labels)
#   print(labels.shape)
#   print(dataset)
#   print(dataset.shape)
#   items = random.sample(range(len(labels)), 8)
#   for i, item in enumerate(items):
#     print(item)
#     plt.subplot(2, 4, i+1)
#     plt.axis('off')
#     plt.title(pretty_labels[labels[item]])
#     plt.imshow(dataset[item],interpolation='nearest')
#     plt.show()
# disp_sample_dataset(train_dataset, train_labels)

pickle_file = 'GENKI4K.pickle'

try:
  f = open(pickle_file, 'wb')
  save = {
    'train_dataset': train_dataset,
    'train_labels': train_labels,
    'valid_dataset': valid_dataset,
    'valid_labels': valid_labels,
    'test_dataset': test_dataset,
    'test_labels': test_labels,
    }
  pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
  f.close()
except Exception as e:
  print('Unable to save data to', pickle_file, ':', e)
  raise

statinfo = os.stat(pickle_file)
print('Compressed pickle size:', statinfo.st_size)

#################################################################

num_labels = 2
num_channels = image_depth # = 3 (RGB)
def reformat(dataset, labels):
  dataset = dataset.reshape(
    (-1, image_size, image_size, num_channels)).astype(np.float32)
  labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
  return dataset, labels
train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)


def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

batch_size = 16
patch_size = 3
drop_out = 0.5

graph = tf.Graph()

with graph.as_default():

  # Input data.
  tf_train_dataset = tf.placeholder(
    tf.float32, shape=(batch_size, image_size, image_size, num_channels))
  tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
  tf_valid_dataset = tf.constant(valid_dataset)
  tf_test_dataset = tf.constant(test_dataset)
  global_step = tf.Variable(0)
  
  # Variables
  layer1_1weights = tf.Variable(tf.truncated_normal(
      [3, 3, 3, 32], stddev=0.1)) 
  layer1_1biases = tf.Variable(tf.zeros([32]))

  layer1_2weights = tf.Variable(tf.truncated_normal(
      [3, 3, 32, 32], stddev=0.1))
  layer1_2biases = tf.Variable(tf.constant(1.0, shape=[32]))
  
  layer2_1weights = tf.Variable(tf.truncated_normal(
      [3, 3, 32, 64], stddev=0.1))
  layer2_1biases = tf.Variable(tf.constant(1.0, shape=[64]))

  layer2_2weights = tf.Variable(tf.truncated_normal(
      [3, 3, 64, 64], stddev=0.1))
  layer2_2biases = tf.Variable(tf.constant(1.0, shape=[64]))

  layer3_1weights = tf.Variable(tf.truncated_normal(
      [3, 3, 64, 128], stddev=0.1))
  layer3_1biases = tf.Variable(tf.constant(1.0, shape=[128]))

  layer3_2weights = tf.Variable(tf.truncated_normal(
      [3, 3, 128, 128], stddev=0.1))
  layer3_2biases = tf.Variable(tf.constant(1.0, shape=[128]))

  layer3_3weights = tf.Variable(tf.truncated_normal(
      [3, 3, 128, 128], stddev=0.1))
  layer3_3biases = tf.Variable(tf.constant(1.0, shape=[128]))

  layer4_1weights = tf.Variable(tf.truncated_normal(
      [3, 3, 128, 256], stddev=0.1))
  layer4_1biases = tf.Variable(tf.constant(1.0, shape=[256]))

  layer4_2weights = tf.Variable(tf.truncated_normal(
      [3, 3, 256, 256], stddev=0.1))
  layer4_2biases = tf.Variable(tf.constant(1.0, shape=[256]))

  layer4_3weights = tf.Variable(tf.truncated_normal(
      [3, 3, 256, 256], stddev=0.1))
  layer4_3biases = tf.Variable(tf.constant(1.0, shape=[256]))

  # big_shape = image_size // 4 * image_size // 4 * image_size // 4 * image_size // 4 * 512
  big_shape = 4096

  fc1w = tf.Variable(tf.truncated_normal(
      [big_shape, 4096], dtype=tf.float32, stddev=0.1))
  fc1b = tf.Variable(tf.constant(1.0, shape=[4096], dtype=tf.float32))

#   fc2w = tf.Variable(tf.truncated_normal(
#     [4096, 4096], dtype=tf.float32, stddev=0.1))
#   fc2b = tf.Variable(tf.constant(1.0, shape=[4096], dtype=tf.float32))

#   fc3w = tf.Variable(tf.truncated_normal(
#       [4096, 1000], dtype=tf.float32, stddev=0.1))
#   fc3b = tf.Variable(tf.constant(1.0, shape=[1000], dtype=tf.float32))
  
  fc4w = tf.Variable(tf.truncated_normal(
      [4096, 2], dtype=tf.float32, stddev=0.1))
  fc4b = tf.Variable(tf.constant(1.0, shape=[2], dtype=tf.float32))
  
  # Model.
  def model(data, keep_prob):
    # conv1
    conv1_1 = tf.nn.conv2d(data, layer1_1weights, [1,1,1,1], padding='SAME')    
    bias1_1 = tf.nn.relu(conv1_1 + layer1_1biases)

    conv1_2 = tf.nn.conv2d(bias1_1, layer1_2weights, [1,1,1,1], padding='SAME')    
    bias1_2 = tf.nn.relu(conv1_2 + layer1_2biases)
    
    pool1 = tf.nn.max_pool(bias1_2, [1,2,2,1], [1,2,2,1], padding='SAME')

    # conv2
    conv2_1 = tf.nn.conv2d(pool1, layer2_1weights, [1,1,1,1], padding='SAME')    
    bias2_1 = tf.nn.relu(conv2_1 + layer2_1biases)

    conv2_2 = tf.nn.conv2d(bias2_1, layer2_2weights, [1,1,1,1], padding='SAME')    
    bias2_2 = tf.nn.relu(conv2_2 + layer2_2biases)
    
    pool2 = tf.nn.max_pool(bias2_2, [1,2,2,1], [1,2,2,1], padding='SAME')

    # conv3
    conv3_1 = tf.nn.conv2d(pool2, layer3_1weights, [1,1,1,1], padding='SAME')    
    bias3_1 = tf.nn.relu(conv3_1 + layer3_1biases)

    conv3_2 = tf.nn.conv2d(bias3_1, layer3_2weights, [1,1,1,1], padding='SAME')    
    bias3_2 = tf.nn.relu(conv3_2 + layer3_2biases)

    conv3_3 = tf.nn.conv2d(bias3_2, layer3_3weights, [1,1,1,1], padding='SAME')    
    bias3_3 = tf.nn.relu(conv3_3 + layer3_3biases)
    
    pool3 = tf.nn.max_pool(bias3_3, [1,2,2,1], [1,2,2,1], padding='SAME')

    # conv4
    conv4_1 = tf.nn.conv2d(pool3, layer4_1weights, [1,1,1,1], padding='SAME')    
    bias4_1 = tf.nn.relu(conv4_1 + layer4_1biases)

    conv4_2 = tf.nn.conv2d(bias4_1, layer4_2weights, [1,1,1,1], padding='SAME')    
    bias4_2 = tf.nn.relu(conv4_2 + layer4_2biases)

    conv4_3 = tf.nn.conv2d(bias4_2, layer4_3weights, [1,1,1,1], padding='SAME')    
    bias4_3 = tf.nn.relu(conv4_3 + layer4_3biases)
    
    pool4 = tf.nn.max_pool(bias4_3, [1,2,2,1], [1,2,2,1], padding='SAME')
    
    # shape = int(np.prod(pool4.get_shape()[1:]))
    shape = int(np.prod(pool4.get_shape()[1:]))

    # fully-connected layer
    # fc1
    # fc1w = tf.Variable(tf.truncated_normal(
    #   [shape, 4096], dtype=tf.float32, stddev=0.1))
    # fc1b = tf.Variable(tf.constant(1.0, shape=[4096], dtype=tf.float32))
    # pool4_flat = tf.reshape(pool4, [-1, shape])    
    # fc1 = tf.nn.relu(tf.matmul(pool4_flat, fc1w) + fc1b)
    pool3_flat = tf.reshape(pool4, [-1, shape])    
    fc1 = tf.nn.relu(tf.matmul(pool3_flat, fc1w) + fc1b)
    drop1 = tf.nn.dropout(fc1, keep_prob)

    # fc2
    # fc2w = tf.Variable(tf.truncated_normal(
    #   [4096, 4096], dtype=tf.float32, stddev=0.1))
    # fc2b = tf.Variable(tf.constant(1.0, shape=[4096], dtype=tf.float32))
#     fc2 = tf.nn.relu(tf.matmul(drop1, fc2w) + fc2b)
#     drop2 = tf.nn.dropout(fc2, keep_prob)

    # fc3
    # fc3w = tf.Variable(tf.truncated_normal(
    #   [4096, 1000], dtype=tf.float32, stddev=0.1))
    # fc3b = tf.Variable(tf.constant(1.0, shape=[1000], dtype=tf.float32))
    # fc3 = tf.nn.relu(tf.matmul(fc2, fc3w) + fc3b)
#     fc3 = tf.nn.relu(tf.matmul(drop1, fc3w) + fc3b)
#     drop3 = tf.nn.dropout(fc3, keep_prob)

    # fc4
    # fc4w = tf.Variable(tf.truncated_normal(
    #   [1000, 2], dtype=tf.float32, stddev=0.1))
    # fc4b = tf.Variable(tf.constant(1.0, shape=[2], dtype=tf.float32))
    return tf.matmul(drop1, fc4w) + fc4b
  
  # Training computation.
  logits = model(tf_train_dataset, drop_out)
  loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))
    
  # Optimizer.
  learning_rate = tf.train.exponential_decay(1e-5, global_step, 1000, 0.85, staircase=True)
  optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
  
  # Predictions for the training, validation, and test data.
  train_prediction = tf.nn.softmax(logits)
  valid_prediction = tf.nn.softmax(model(tf_valid_dataset, 1.0))
  test_prediction = tf.nn.softmax(model(tf_test_dataset, 1.0))

num_steps = 20001

with tf.Session(graph=graph) as session:
  tf.initialize_all_variables().run()
  print('Initialized')
  for step in range(num_steps):
    offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
    batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
    batch_labels = train_labels[offset:(offset + batch_size), :]
    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
    _, l, predictions = session.run(
      [optimizer, loss, train_prediction], feed_dict=feed_dict)
    if (step % 50 == 0):
      print('Minibatch loss at step %d: %f' % (step, l))
      print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
      print('Validation accuracy: %.1f%%' % accuracy(
        valid_prediction.eval(), valid_labels))
  print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))