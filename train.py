import numpy as np
import tensorflow as tf

import argparse
import time
import os
import cPickle

from utils import SketchLoader
from model import Model

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--rnn_size', type=int, default=256,
                     help='size of RNN hidden state')
  parser.add_argument('--num_layers', type=int, default=2,
                     help='number of layers in the RNN')
  parser.add_argument('--model', type=str, default='lstm',
                     help='rnn, gru, or lstm')
  parser.add_argument('--batch_size', type=int, default=100,
                     help='minibatch size')
  parser.add_argument('--seq_length', type=int, default=300,
                     help='RNN sequence length')
  parser.add_argument('--num_epochs', type=int, default=500,
                     help='number of epochs')
  parser.add_argument('--save_every', type=int, default=250,
                     help='save frequency')
  parser.add_argument('--grad_clip', type=float, default=5.0,
                     help='clip gradients at this value')
  parser.add_argument('--learning_rate', type=float, default=0.005,
                     help='learning rate')
  parser.add_argument('--decay_rate', type=float, default=0.99,
                     help='decay rate for rmsprop')
  parser.add_argument('--num_mixture', type=int, default=24,
                     help='number of gaussian mixtures')
  parser.add_argument('--data_scale', type=float, default=15.0,
                     help='factor to scale raw data down by')
  parser.add_argument('--keep_prob', type=float, default=0.8,
                     help='dropout keep probability')
  parser.add_argument('--stroke_importance_factor', type=float, default=200.0,
                     help='relative importance of pen status over mdn coordinate accuracy')
  parser.add_argument('--dataset_name', type=str, default="kanji",
                     help='name of directory containing training data')
  args = parser.parse_args()
  train(args)

def train(args):
  data_loader = SketchLoader(args.batch_size, args.seq_length, args.data_scale, args.dataset_name)

  dirname = os.path.join('save', args.dataset_name)
  if not os.path.exists(dirname):
    os.makedirs(dirname)

  with open(os.path.join(dirname, 'config.pkl'), 'w') as f:
      cPickle.dump(args, f)

  model = Model(args)

  b_processed = 0

  with tf.Session() as sess:

    tf.initialize_all_variables().run()
    saver = tf.train.Saver(tf.all_variables())

    # load previously trained model if appilcable
    ckpt = tf.train.get_checkpoint_state(os.path.join('save', args.dataset_name))
    if ckpt:
      print "loading last model: ",ckpt.model_checkpoint_path
      saver.restore(sess, ckpt.model_checkpoint_path)

    def save_model():
      checkpoint_path = os.path.join('save', args.dataset_name, 'model.ckpt')
      saver.save(sess, checkpoint_path, global_step = b_processed)
      print "model saved to {}".format(checkpoint_path)

    for e in xrange(args.num_epochs):
      sess.run(tf.assign(model.lr, args.learning_rate * (args.decay_rate ** e)))
      data_loader.reset_index_pointer()
      state = model.initial_state.eval()
      while data_loader.epoch_finished == False:
        start = time.time()
        input_data, target_data = data_loader.next_batch()
        feed = {model.input_data: input_data, model.target_data: target_data, model.initial_state: state}
        train_loss, shape_loss, pen_loss, state, _ = sess.run([model.cost, model.cost_shape, model.cost_pen, model.final_state, model.train_op], feed)
        end = time.time()
        b_processed += 1
        print "{}/{} (epoch {} batch {}), cost = {:.2f} ({:.2f}+{:.4f}), time/batch = {:.2f}" \
          .format(data_loader.pointer + e * data_loader.num_samples,
            args.num_epochs * data_loader.num_samples,
            e, b_processed ,train_loss, shape_loss, pen_loss, end - start)
        # assert( train_loss != np.NaN or train_loss != np.Inf) # doesn't work.
        assert( train_loss < 30000) # if dodgy loss, exit w/ error.
        if (b_processed) % args.save_every == 0 and ((b_processed) > 0):
          save_model()
    save_model()

if __name__ == '__main__':
  main()


