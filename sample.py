import numpy as np
import tensorflow as tf

import time
import os
import cPickle
import argparse

from utils import *
from model import Model
import random

import svgwrite
from IPython.display import SVG, display

# main code (not in a main function since I want to run this script in IPython as well).
def in_ipython():
  try:
    __IPYTHON__
  except NameError:
    return False
  else:
    return True


parser = argparse.ArgumentParser()
parser.add_argument('--filename', type=str, default='output',
                   help='filename of .svg file to output, without .svg')
parser.add_argument('--sample_length', type=int, default=600,
                   help='number of strokes to sample')
parser.add_argument('--picture_size', type=float, default=160,
                   help='a centered svg will be generated of this size')
parser.add_argument('--scale_factor', type=float, default=1,
                   help='factor to scale down by for svg output.  smaller means bigger output')
parser.add_argument('--num_picture', type=int, default=20,
                   help='number of pictures to generate')
parser.add_argument('--num_col', type=int, default=5,
                   help='if num_picture > 1, how many pictures per row?')
parser.add_argument('--dataset_name', type=str, default="kanji",
                   help='name of directory containing training data')
parser.add_argument('--color_mode', type=int, default=1,
                   help='set to 0 if you are a black and white sort of person...')
parser.add_argument('--stroke_width', type=float, default=2.0,
                   help='thickness of pen lines')
parser.add_argument('--temperature', type=float, default=0.1,
                   help='sampling temperature')
sample_args = parser.parse_args()

color_mode = True
if sample_args.color_mode == 0:
  color_mode = False


with open(os.path.join('save', sample_args.dataset_name, 'config.pkl')) as f: # future
  saved_args = cPickle.load(f)

model = Model(saved_args, True)
sess = tf.InteractiveSession()
saver = tf.train.Saver(tf.all_variables())

ckpt = tf.train.get_checkpoint_state(os.path.join('save', sample_args.dataset_name))
print "loading model: ",ckpt.model_checkpoint_path

saver.restore(sess, ckpt.model_checkpoint_path)

def draw_sketch_array(strokes_array, svg_only = False):
  draw_stroke_color_array(strokes_array, factor=sample_args.scale_factor, maxcol = sample_args.num_col, svg_filename = sample_args.filename+'.svg', stroke_width = sample_args.stroke_width, block_size = sample_args.picture_size, svg_only = svg_only, color_mode = color_mode)

def sample_sketches(min_size_ratio = 0.0, max_size_ratio = 0.8, min_num_stroke = 4, max_num_stroke=22, svg_only = True):
  N = sample_args.num_picture
  frame_size = float(sample_args.picture_size)
  max_size = frame_size * max_size_ratio
  min_size = frame_size * min_size_ratio
  count = 0
  sketch_list = []
  param_list = []

  temp_mixture = sample_args.temperature
  temp_pen = sample_args.temperature

  while count < N:
    #print "attempting to generate picture #", count
    print '.',
    [strokes, params] = model.sample(sess, sample_args.sample_length, temp_mixture, temp_pen, stop_if_eoc = True)
    [sx, sy, num_stroke, num_char, _] = strokes.sum(0)
    if num_stroke < min_num_stroke or num_char == 0 or num_stroke > max_num_stroke:
      #print "num_stroke ", num_stroke, " num_char ", num_char
      continue
    [sx, sy, sizex, sizey] = calculate_start_point(strokes)
    if sizex > max_size or sizey > max_size:
      #print "sizex ", sizex, " sizey ", sizey
      continue
    if sizex < min_size or sizey < min_size:
      #print "sizex ", sizex, " sizey ", sizey
      continue
    # success
    print count+1,"/",N
    count += 1
    sketch_list.append(strokes)
    param_list.append(params)
  # draw the pics
  draw_sketch_array(sketch_list, svg_only = svg_only)
  return sketch_list, param_list

if __name__ == '__main__':
  ipython_mode = in_ipython()
  if ipython_mode:
    print "IPython detected"
  else:
    print "Console mode"
  [strokes, params] = sample_sketches(svg_only = not ipython_mode)


