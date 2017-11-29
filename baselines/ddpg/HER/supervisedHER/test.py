from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.examples.tutorials.mnist import input_data
import tabular_logger as tlogger
import tensorflow as tf
import numpy as np
import argparse
import random
import time
import sys
import os

def max_pool_2x2(x):
  """max_pool_2x2 downsamples a feature map by 2X."""
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def train(args):
  if args.conv:
    log_dir = '{}/hindsight{}/batch{}Conv'.format(args.log_dir, args.batch_size, args.hindsight)
  else:
    log_dir = '{}/hindsight{}/batch{}/FC'.format(args.log_dir, args.batch_size, args.hindsight)
  tlogger.start(log_dir)
  for k, v in args.__dict__.items():
    tlogger.log('{}: {}'.format(k, v))
  model = HER(args)

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    M = 55000

    tstart = time.time()
    for i in range(args.num_iterations):
      start = time.time()
      x, y = model.mnist.train.next_batch(args.batch_size)
      hindsight_x = model.hindsight_lookup(y)
      _, train_accuracy, train_cross, train_loss = sess.run(
        [model.train_step, model.accuracy, model.cross_entropy, 
        model.loss], feed_dict={model.x: x, model.y_: y, 
        model.hindsight_x: hindsight_x})

      if i % 1000 == 0:
        # train_writer.add_summary(train_summary, i)
        tlogger.log('********** Iteration {} **********'.format(i))
        tlogger.record_tabular("train_cross", train_cross)
        tlogger.record_tabular("train_loss", train_loss)
        tlogger.record_tabular("train_acc", train_accuracy)

        xs, ys = model.mnist.test.images, model.mnist.test.labels
        test_accuracy, test_loss = sess.run([model.accuracy, model.cross_entropy], 
          feed_dict={model.x: xs, model.y_: ys, model.hindsight_x: hindsight_x})
        tlogger.record_tabular("test_loss", test_loss)
        tlogger.record_tabular("test_acc", test_accuracy)
        tlogger.record_tabular("TimeElapsed", time.time() - tstart)
        tlogger.dump_tabular()
    tlogger.stop()

class HER():
  def __init__(self, args):
    self.args = args
    self.data_dir = args.data_dir
    self.batch_size = args.batch_size
    self.log_dir = args.log_dir
    self.learning_rate = args.learning_rate
    self.mnist = input_data.read_data_sets('../data/mnist', one_hot=True)
    self.hindsight_images = self.mnist.train.images
    self.hindsight_y = np.argmax(self.mnist.train.labels, axis=1)
    labels = [i for i in range(10)]
    self.hindsight_dict = dict((key,[]) for key in labels)
    for index, label in enumerate(self.hindsight_y):
      self.hindsight_dict[label].append(index)

    with tf.variable_scope("model"):
      self.scope = 'HER'
      self.template = self.make_template()
      self.x = tf.placeholder(tf.float32, [None, 784], name='x-input')
      self.hindsight_x = tf.placeholder(tf.float32, [None, 784], name='x-input')
      self.y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')
      logits = self.template(self.x)
      self.cross_entropy = tf.reduce_mean(
          tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=logits))
      if args.hindsight:
        hindsight_logits = self.template(self.hindsight_x)
        self.hindsight_loss = tf.sqrt(tf.reduce_mean(
          tf.square(tf.subtract(logits, hindsight_logits))))
        self.loss = self.cross_entropy + self.hindsight_loss
      else:
        self.loss = self.cross_entropy

      self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(
          self.loss)
      correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(self.y_, 1))
      self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  def make_template(self):
    def fc_spec(x):
      if self.args.conv:
        x = tf.reshape(x, [-1,28,28,1])
        x = tf.contrib.layers.conv2d(x, 32, [5,5], scope='conv1')
        x = max_pool_2x2(x)
        x = tf.contrib.layers.conv2d(x, 64, [5,5], scope='conv2')
        x = max_pool_2x2(x)
        x = tf.reshape(x, [-1, 7*7*64])
        x = tf.layers.dense(x, 1024, activation=tf.nn.relu, name='fc1')
        logits = tf.layers.dense(x, 10, name='fc2')
      else:
        x = tf.layers.dense(x, 512, activation=tf.nn.relu, name='fc1')
        x = tf.layers.dense(x, 512, activation=tf.nn.relu, name='fc2')
        logits = tf.layers.dense(x, 10, name='fc3')
      return logits
    return tf.make_template(self.scope, fc_spec)

  def hindsight_lookup(self, labels):
    # Given a batch of one-hot label.
    y = np.argmax(labels, axis=1)
    indices = [random.choice(self.hindsight_dict[label]) for label in y]
    hindsight_x = np.array([self.hindsight_images[index,:] for index in indices])
    return hindsight_x

if __name__ == '__main__':
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--data_dir', type=str, default='../data/mnist',
                      help='data file dir')
  parser.add_argument('--log_dir', type=str, default='log/',
                      help='log file dir')
  parser.add_argument('--learning_rate', type=float, default=1e-4,
                      help='learning rate')
  parser.add_argument('--batch_size', type=int, default=128,
                      help='minibatch size')
  parser.add_argument('--num_iterations', type=int, default=30000,
                      help='number of iterations')
  parser.add_argument('--hindsight', action='store_true', default=False)
  parser.add_argument('--conv', action='store_true', default=False)
  parser.add_argument('--dropout', action='store_true', default=False)
  args = parser.parse_args()
  train(args)