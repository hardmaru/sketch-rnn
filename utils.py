import os
import cPickle
import numpy as np
import xml.etree.ElementTree as ET
import random
import svgwrite
from IPython.display import SVG, display
from svg.path import Path, Line, Arc, CubicBezier, QuadraticBezier, parse_path

def calculate_start_point(data, factor=1.0, block_size = 200):
  # will try to center the sketch to the middle of the block
  # determines maxx, minx, maxy, miny
  sx = 0
  sy = 0
  maxx = 0
  minx = 0
  maxy = 0
  miny = 0
  for i in xrange(len(data)):
    sx += round(float(data[i, 0])*factor, 3)
    sy += round(float(data[i, 1])*factor, 3)
    maxx = max(maxx, sx)
    minx = min(minx, sx)
    maxy = max(maxy, sy)
    miny = min(miny, sy)

  abs_x = block_size/2-(maxx-minx)/2-minx
  abs_y = block_size/2-(maxy-miny)/2-miny

  return abs_x, abs_y, (maxx-minx), (maxy-miny)

def draw_stroke_color_array(data, factor=1, svg_filename = 'sample.svg', stroke_width = 1, block_size = 200, maxcol = 5, svg_only = False, color_mode = True):

  num_char = len(data)

  if num_char < 1:
    return

  max_color_intensity = 225

  numrow = np.ceil(float(num_char)/float(maxcol))
  dwg = svgwrite.Drawing(svg_filename, size=(block_size*(min(num_char, maxcol)), block_size*numrow))
  dwg.add(dwg.rect(insert=(0, 0), size=(block_size*(min(num_char, maxcol)), block_size*numrow),fill='white'))

  the_color = "rgb("+str(random.randint(0, max_color_intensity))+","+str(int(random.randint(0, max_color_intensity)))+","+str(int(random.randint(0, max_color_intensity)))+")"

  for j in xrange(len(data)):

    lift_pen = 0
    #end_of_char = 0
    cdata = data[j]
    abs_x, abs_y, size_x, size_y = calculate_start_point(cdata, factor, block_size)
    abs_x += (j % maxcol) * block_size
    abs_y += (j / maxcol) * block_size

    for i in xrange(len(cdata)):

      x = round(float(cdata[i,0])*factor, 3)
      y = round(float(cdata[i,1])*factor, 3)

      prev_x = round(abs_x, 3)
      prev_y = round(abs_y, 3)

      abs_x += x
      abs_y += y

      if (lift_pen == 1):
        p = "M "+str(abs_x)+","+str(abs_y)+" "
        the_color = "rgb("+str(random.randint(0, max_color_intensity))+","+str(int(random.randint(0, max_color_intensity)))+","+str(int(random.randint(0, max_color_intensity)))+")"
      else:
        p = "M "+str(prev_x)+","+str(prev_y)+" L "+str(abs_x)+","+str(abs_y)+" "

      lift_pen = max(cdata[i, 2], cdata[i, 3]) # lift pen if both eos or eoc
      #end_of_char = cdata[i, 3] # not used for now.

      if color_mode == False:
        the_color = "#000"

      dwg.add(dwg.path(p).stroke(the_color,stroke_width).fill(the_color)) #, opacity=round(random.random()*0.5+0.5, 3)

  dwg.save()
  if svg_only == False:
    display(SVG(dwg.tostring()))

def draw_stroke_color(data, factor=1, svg_filename = 'sample.svg', stroke_width = 1, block_size = 200, maxcol = 5, svg_only = False, color_mode = True):

  def split_sketch(data):
    # split a sketch with many eoc into an array of sketches, each with just one eoc at the end.
    # ignores last stub with no eoc.
    counter = 0
    result = []
    for i in xrange(len(data)):
      eoc = data[i, 3]
      if eoc > 0:
        result.append(data[counter:i+1])
        counter = i+1
    #if (counter < len(data)): # ignore the rest
    #  result.append(data[counter:])
    return result

  data = np.array(data, dtype=np.float32)
  data = split_sketch(data)
  draw_stroke_color_array(data, factor, svg_filename, stroke_width, block_size, maxcol, svg_only, color_mode)

class SketchLoader():
  def __init__(self, batch_size=50, seq_length=300, scale_factor = 1.0, data_filename = "kanji"):
    self.data_dir = "./data"
    self.batch_size = batch_size
    self.seq_length = seq_length
    self.scale_factor = scale_factor # divide data by this factor

    data_file = os.path.join(self.data_dir, data_filename+".cpkl")
    raw_data_dir = os.path.join(self.data_dir, data_filename)

    if not (os.path.exists(data_file)) :
        print "creating training data cpkl file from raw source"
        self.length_data = self.preprocess(raw_data_dir, data_file)

    self.load_preprocessed(data_file)
    self.num_samples = len(self.raw_data)
    self.index = range(self.num_samples) # this list will be randomized later.
    self.reset_index_pointer()

  def preprocess(self, data_dir, data_file):
    # create data file from raw xml files from iam handwriting source.
    len_data = []
    def cubicbezier(x0, y0, x1, y1, x2, y2, x3, y3, n=20):
      # from http://rosettacode.org/wiki/Bitmap/B%C3%A9zier_curves/Cubic
      pts = []
      for i in range(n+1):
        t = float(i) / float(n)
        a = (1. - t)**3
        b = 3. * t * (1. - t)**2
        c = 3.0 * t**2 * (1.0 - t)
        d = t**3

        x = float(a * x0 + b * x1 + c * x2 + d * x3)
        y = float(a * y0 + b * y1 + c * y2 + d * y3)
        pts.append( (x, y) )
      return pts

    def get_path_strings(svgfile):
      tree = ET.parse(svgfile)
      p = []
      for elem in tree.iter():
        if elem.attrib.has_key('d'):
          p.append(elem.attrib['d'])
      return p

    def build_lines(svgfile, line_length_threshold = 10.0, min_points_per_path = 1, max_points_per_path = 3):
      # we don't draw lines less than line_length_threshold
      path_strings = get_path_strings(svgfile)

      lines = []

      for path_string in path_strings:
        full_path = parse_path(path_string)
        for i in range(len(full_path)):
          p = full_path[i]
          if type(p) != Line and type(p) != CubicBezier:
            print "encountered an element that is not just a line or bezier "
            print "type: ",type(p)
            print p
          else:
            x_start = p.start.real
            y_start = p.start.imag
            x_end = p.end.real
            y_end = p.end.imag
            line_length = np.sqrt((x_end-x_start)*(x_end-x_start)+(y_end-y_start)*(y_end-y_start))
            len_data.append(line_length)
            points = []
            if type(p) == CubicBezier:
              x_con1 = p.control1.real
              y_con1 = p.control1.imag
              x_con2 = p.control2.real
              y_con2 = p.control2.imag
              n_points = int(line_length / line_length_threshold)+1
              n_points = max(n_points, min_points_per_path)
              n_points = min(n_points, max_points_per_path)
              points = cubicbezier(x_start, y_start, x_con1, y_con1, x_con2, y_con2, x_end, y_end, n_points)
            else:
              points = [(x_start, y_start), (x_end, y_end)]
            if i == 0: # only append the starting point for svg
              lines.append([points[0][0], points[0][1], 0, 0]) # put eoc to be zero
            for j in range(1, len(points)):
              eos = 0
              if j == len(points)-1 and i == len(full_path)-1:
                eos = 1
              lines.append([points[j][0], points[j][1], eos, 0]) # put eoc to be zero
      lines = np.array(lines, dtype=np.float32)
      # make it relative moves
      lines[1:,0:2] -= lines[0:-1,0:2]
      lines[-1,3] = 1 # end of character
      lines[0] = [0, 0, 0, 0] # start at origin
      return lines[1:]

    # build the list of xml files
    filelist = []
    # Set the directory you want to start from
    rootDir = data_dir
    for dirName, subdirList, fileList in os.walk(rootDir):
      #print('Found directory: %s' % dirName)
      for fname in fileList:
        #print('\t%s' % fname)
        filelist.append(dirName+"/"+fname)

    # build stroke database of every xml file inside iam database
    sketch = []
    for i in range(len(filelist)):
      if (filelist[i][-3:] == 'svg'):
        print 'processing '+filelist[i]
        sketch.append(build_lines(filelist[i]))

    f = open(data_file,"wb")
    cPickle.dump(sketch, f, protocol=2)
    f.close()
    return len_data

  def load_preprocessed(self, data_file):
    f = open(data_file,"rb")
    self.raw_data = cPickle.load(f)
    # scale the data here, rather than at the data construction (since scaling may change)
    for data in self.raw_data:
      data[:,0:2] /= self.scale_factor
    f.close()

  def next_batch(self):
    # returns a set of batches, but the constraint is that the start of each input data batch
    # is the start of a new character (although the end of a batch doesn't have to be end of a character)

    def next_seq(n):
      result = np.zeros((n, 5), dtype=np.float32) # x, y, [eos, eoc, cont] tokens
      #result[0, 2:4] = 1 # set eos and eoc to true for first point
      # experimental line below, put a random factor between 70-130% to generate more examples
      rand_scale_factor_x = np.random.rand()*0.6+0.7
      rand_scale_factor_y = np.random.rand()*0.6+0.7
      idx = 0
      data = self.current_data()
      for i in xrange(n):
        result[i, 0:4] = data[idx] # eoc = 0.0
        result[i, 4] = 1 # continue on stroke
        if (result[i, 2] > 0 or result[i, 3] > 0):
          result[i, 4] = 0
        idx += 1
        if (idx >= len(data)-1): # skip to next sketch example next time and mark eoc
          result[i, 4] = 0
          result[i, 3] = 1
          result[i, 2] = 0 # overrides end of stroke one-hot
          idx = 0
          self.tick_index_pointer()
          data = self.current_data()
        assert(result[i, 2:5].sum() == 1)
      self.tick_index_pointer() # needed if seq_length is less than last data.
      result[:, 0] *= rand_scale_factor_x
      result[:, 1] *= rand_scale_factor_y
      return result

    skip_length = self.seq_length+1

    batch = []

    for i in xrange(self.batch_size):
      seq = next_seq(skip_length)
      batch.append(seq)

    batch = np.array(batch, dtype=np.float32)

    return batch[:,0:-1], batch[:, 1:]

  def current_data(self):
    return self.raw_data[self.index[self.pointer]]

  def tick_index_pointer(self):
    self.pointer += 1
    if (self.pointer >= len(self.raw_data)):
      self.pointer = 0
      self.epoch_finished = True

  def reset_index_pointer(self):
    # randomize order for the raw list in the next go.
    self.pointer = 0
    self.epoch_finished = False
    self.index = np.random.permutation(self.index)


