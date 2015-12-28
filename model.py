import tensorflow as tf
from tensorflow.models.rnn import rnn_cell

import numpy as np
import random

class Model():
  def __init__(self, args, infer=False):
    if infer:
      args.batch_size = 1
      args.seq_length = 1
    self.args = args

    if args.model == 'rnn':
      cell_fn = rnn_cell.BasicRNNCell
    elif args.model == 'gru':
      cell_fn = rnn_cell.GRUCell
    elif args.model == 'lstm':
      cell_fn = rnn_cell.BasicLSTMCell
    else:
      raise Exception("model type not supported: {}".format(args.model))

    cell = cell_fn(args.rnn_size)

    cell = rnn_cell.MultiRNNCell([cell] * args.num_layers)

    if (infer == False and args.keep_prob < 1): # training mode
      cell = rnn_cell.DropoutWrapper(cell, output_keep_prob = args.keep_prob)

    self.cell = cell

    self.input_data = tf.placeholder(dtype=tf.float32, shape=[args.batch_size, args.seq_length, 5])
    self.target_data = tf.placeholder(dtype=tf.float32, shape=[args.batch_size, args.seq_length, 5])
    self.initial_state = cell.zero_state(batch_size=args.batch_size, dtype=tf.float32)

    self.num_mixture = args.num_mixture
    NOUT = 3 + self.num_mixture * 6 # [end_of_stroke + end_of_char, continue_with_stroke] + prob + 2*(mu + sig) + corr

    with tf.variable_scope('rnn_mdn'):
      output_w = tf.get_variable("output_w", [args.rnn_size, NOUT])
      output_b = tf.get_variable("output_b", [NOUT])

    inputs = tf.split(1, args.seq_length, self.input_data)
    inputs = [tf.squeeze(input_, [1]) for input_ in inputs]

    self.initial_input = np.zeros((args.batch_size, 5), dtype=np.float32)
    self.initial_input[:,4] = 1.0 # initially, the pen is down.
    self.initial_input = tf.constant(self.initial_input)

    def tfrepeat(a, repeats):
      num_row = a.get_shape()[0].value
      num_col = a.get_shape()[1].value
      assert(num_col == 1)
      result = [a for i in range(repeats)]
      result = tf.concat(0, result)
      result = tf.reshape(result, [repeats, num_row])
      result = tf.transpose(result)
      return result

    def custom_rnn_autodecoder(decoder_inputs, initial_input, initial_state, cell, scope=None):
      # customized rnn_decoder for the task of dealing with end of character
      with tf.variable_scope(scope or "rnn_decoder"):
        states = [initial_state]
        outputs = []
        prev = None

        for i in xrange(len(decoder_inputs)):
          inp = decoder_inputs[i]
          if i > 0:
            tf.get_variable_scope().reuse_variables()
          output, new_state = cell(inp, states[-1])

          num_batches = self.args.batch_size # new_state.get_shape()[0].value
          num_state = new_state.get_shape()[1].value

          # if the input has an end-of-character signal, have to zero out the state

          #to do:  test this code.

          eoc_detection = inp[:,3]
          eoc_detection = tf.reshape(eoc_detection, [num_batches, 1])

          eoc_detection_state = tfrepeat(eoc_detection, num_state)

          eoc_detection_state = tf.greater(eoc_detection_state, tf.zeros_like(eoc_detection_state, dtype=tf.float32))

          new_state = tf.select(eoc_detection_state, initial_state, new_state)

          outputs.append(output)
          states.append(new_state)
      return outputs, states

    outputs, states = custom_rnn_autodecoder(inputs, self.initial_input, self.initial_state, cell, scope='rnn_mdn')
    output = tf.reshape(tf.concat(1, outputs), [-1, args.rnn_size])
    output = tf.nn.xw_plus_b(output, output_w, output_b)
    self.final_state = states[-1]

    # reshape target data so that it is compatible with prediction shape
    flat_target_data = tf.reshape(self.target_data,[-1, 5])
    [x1_data, x2_data, eos_data, eoc_data, cont_data] = tf.split(1, 5, flat_target_data)
    pen_data = tf.concat(1, [eos_data, eoc_data, cont_data])

    # long method:
    #flat_target_data = tf.split(1, args.seq_length, self.target_data)
    #flat_target_data = [tf.squeeze(flat_target_data_, [1]) for flat_target_data_ in flat_target_data]
    #flat_target_data = tf.reshape(tf.concat(1, flat_target_data), [-1, 3])

    def tf_2d_normal(x1, x2, mu1, mu2, s1, s2, rho):
      # eq # 24 and 25 of http://arxiv.org/abs/1308.0850
      norm1 = tf.sub(x1, mu1)
      norm2 = tf.sub(x2, mu2)
      s1s2 = tf.mul(s1, s2)
      z = tf.square(tf.div(norm1, s1))+tf.square(tf.div(norm2, s2))-2*tf.div(tf.mul(rho, tf.mul(norm1, norm2)), s1s2)
      negRho = 1-tf.square(rho)
      result = tf.exp(tf.div(-z,2*negRho))
      denom = 2*np.pi*tf.mul(s1s2, tf.sqrt(negRho))
      result = tf.div(result, denom)
      return result

    def get_lossfunc(z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr, z_pen, x1_data, x2_data, pen_data):
      result0 = tf_2d_normal(x1_data, x2_data, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr)
      # implementing eq # 26 of http://arxiv.org/abs/1308.0850
      epsilon = 1e-20
      result1 = tf.mul(result0, z_pi)
      result1 = tf.reduce_sum(result1, 1, keep_dims=True)
      result1 = -tf.log(tf.maximum(result1, 1e-20)) # at the beginning, some errors are exactly zero.
      result_shape = tf.reduce_mean(result1)

      result2 = tf.nn.softmax_cross_entropy_with_logits(z_pen, pen_data)
      pen_data_weighting = pen_data[:, 2]+np.sqrt(self.args.stroke_importance_factor)*pen_data[:, 0]+self.args.stroke_importance_factor*pen_data[:, 1]
      result2 = tf.mul(result2, pen_data_weighting)
      result_pen = tf.reduce_mean(result2)

      result = result_shape + result_pen
      return result, result_shape, result_pen,

    # below is where we need to do MDN splitting of distribution params
    def get_mixture_coef(output):
      # returns the tf slices containing mdn dist params
      # ie, eq 18 -> 23 of http://arxiv.org/abs/1308.0850
      z = output
      z_pen = z[:, 0:3] # end of stroke, end of character/content, continue w/ stroke
      z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr = tf.split(1, 6, z[:, 3:])

      # process output z's into MDN paramters

      # softmax all the pi's:
      max_pi = tf.reduce_max(z_pi, 1, keep_dims=True)
      z_pi = tf.sub(z_pi, max_pi)
      z_pi = tf.exp(z_pi)
      normalize_pi = tf.inv(tf.reduce_sum(z_pi, 1, keep_dims=True))
      z_pi = tf.mul(normalize_pi, z_pi)

      # exponentiate the sigmas and also make corr between -1 and 1.
      z_sigma1 = tf.exp(z_sigma1)
      z_sigma2 = tf.exp(z_sigma2)
      z_corr = tf.tanh(z_corr)

      return [z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr, z_pen]

    [o_pi, o_mu1, o_mu2, o_sigma1, o_sigma2, o_corr, o_pen] = get_mixture_coef(output)

    self.pi = o_pi
    self.mu1 = o_mu1
    self.mu2 = o_mu2
    self.sigma1 = o_sigma1
    self.sigma2 = o_sigma2
    self.corr = o_corr
    self.pen = o_pen # state of the pen

    [lossfunc, loss_shape, loss_pen] = get_lossfunc(o_pi, o_mu1, o_mu2, o_sigma1, o_sigma2, o_corr, o_pen, x1_data, x2_data, pen_data)
    self.cost = lossfunc
    self.cost_shape = loss_shape
    self.cost_pen = loss_pen

    self.lr = tf.Variable(0.01, trainable=False)
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), args.grad_clip)
    optimizer = tf.train.AdamOptimizer(self.lr, epsilon=0.001)
    self.train_op = optimizer.apply_gradients(zip(grads, tvars))


  def sample(self, sess, num=300, temp_mixture=1.0, temp_pen=1.0, stop_if_eoc = False):

    def get_pi_idx(x, pdf):
      N = pdf.size
      accumulate = 0
      for i in range(0, N):
        accumulate += pdf[i]
        if (accumulate >= x):
          return i
      print 'error with sampling ensemble'
      return -1

    def sample_gaussian_2d(mu1, mu2, s1, s2, rho):
      mean = [mu1, mu2]
      cov = [[s1*s1, rho*s1*s2], [rho*s1*s2, s2*s2]]
      x = np.random.multivariate_normal(mean, cov, 1)
      return x[0][0], x[0][1]

    prev_x = np.zeros((1, 1, 5), dtype=np.float32)
    #prev_x[0, 0, 2] = 1 # initially, we want to see beginning of new stroke
    #prev_x[0, 0, 3] = 1 # initially, we want to see beginning of new character/content
    prev_state = sess.run(self.cell.zero_state(self.args.batch_size, tf.float32))

    strokes = np.zeros((num, 5), dtype=np.float32)
    mixture_params = []

    for i in xrange(num):

      feed = {self.input_data: prev_x, self.initial_state:prev_state}

      [o_pi, o_mu1, o_mu2, o_sigma1, o_sigma2, o_corr, o_pen, next_state] = sess.run([self.pi, self.mu1, self.mu2, self.sigma1, self.sigma2, self.corr, self.pen, self.final_state],feed)

      pi_pdf = o_pi[0]
      if i > 1:
        pi_pdf = np.log(pi_pdf) / temp_mixture
        pi_pdf -= pi_pdf.max()
        pi_pdf = np.exp(pi_pdf)
      pi_pdf /= pi_pdf.sum()

      idx = get_pi_idx(random.random(), pi_pdf)

      pen_pdf = o_pen[0]
      if i > 1:
        pi_pdf /= temp_pen # softmax convert to prob
      pen_pdf -= pen_pdf.max()
      pen_pdf = np.exp(pen_pdf)
      pen_pdf /= pen_pdf.sum()

      pen_idx = get_pi_idx(random.random(), pen_pdf)
      eos = 0
      eoc = 0
      cont_state = 0

      if pen_idx == 0:
        eos = 1
      elif pen_idx == 1:
        eoc = 1
      else:
        cont_state = 1

      next_x1, next_x2 = sample_gaussian_2d(o_mu1[0][idx], o_mu2[0][idx], o_sigma1[0][idx], o_sigma2[0][idx], o_corr[0][idx])

      strokes[i,:] = [next_x1, next_x2, eos, eoc, cont_state]

      params = [pi_pdf, o_mu1[0], o_mu2[0], o_sigma1[0], o_sigma2[0], o_corr[0], pen_pdf]
      mixture_params.append(params)

      # early stopping condition
      if (stop_if_eoc and eoc == 1):
        strokes = strokes[0:i+1, :]
        break

      prev_x = np.zeros((1, 1, 5), dtype=np.float32)
      prev_x[0][0] = np.array([next_x1, next_x2, eos, eoc, cont_state], dtype=np.float32)
      prev_state = next_state

    strokes[:,0:2] *= self.args.data_scale
    return strokes, mixture_params


