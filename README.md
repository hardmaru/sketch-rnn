# sketch-rnn

Implementation multi-layer recurrent neural network (RNN, LSTM GRU) used to model and generate sketches stored in .svg vector graphic files.  The methodology used is to combine Mixture Density Networks with a RNN, along with modelling dynamic end-of-stroke and end-of-content probabilities learned from a large corpus of similar .svg files, to generate drawings that is simlar to the vector training data.

See my blog post at [blog.otoro.net](http://blog.otoro.net/2015/12/28/recurrent-net-dreams-up-fake-chinese-characters-in-vector-format-with-tensorflow/) for a detailed description on applying `sketch-rnn`  to learn to generate fake Chinese characters in vector format.

Example Training Sketches (20 randomly chosen out of 11000 [KanjiVG](http://kanjivg.tagaini.net/) dataset):

![Example Training Sketches](https://cdn.rawgit.com/hardmaru/sketch-rnn/master/example/training.svg)

Generated Sketches (Temperature = 0.1):

![Generated Sketches](https://cdn.rawgit.com/hardmaru/sketch-rnn/master/example/output.svg)

# Basic Usage

I tested the implementation on TensorFlow 0.50.  I also used the following libraries to help:

```
svgwrite
IPython.display.SVG
IPython.display.display
xml.etree.ElementTree
argparse
cPickle
svg.path
```

## Loading in Training Data

The training data is located inside the `data` subdirectory.  In this repo, I've included `kanji.cpkl` which is a preprocessed array of KanjiVG characters.

To add a new set of training data, for example, from the [TU Berlin Sketch Database](http://cybertron.cg.tu-berlin.de/eitz/projects/classifysketch/), you have to create a subdirectory, say `tuberlin` inside the `data` directory, and in additionl create a directory of the same name in the `save` directory.  So you end up with `data/tuberlin/` and `save/tuberlin`, where `tuberlin` is defined as a name field for flags in the training and sample programs later on.  `save/tuberlin` will contain the check-pointed trained models later on.

Now, put a large collection .svg files into `data/tuberlin/`.  You can even create subdirectories within `data/tuberlin/` and it will work, as the `SketchLoader` class will scan the entire subdirectory tree.

Currently, `sketch-rnn` only processes `path` elements inside svg files, and within the `path` elements, it only cares about lines and belzier curves at the moment.  I found this sufficient to handle TUBerlin and KanjiVG databases, although it wouldn't be difficult to extent to process the other curve elements, even shape elements in the future.

You can use `utils.py` to play out some random training data after the svg files have been copied in:

```
%run -i utils.py
loader = SketchLoader(data_filename = 'tuberlin')
draw_stroke_color(random.choice(loader.raw_data))
```

![Example Elephant from TU Berlin database](https://cdn.rawgit.com/hardmaru/sketch-rnn/master/example/elephant.svg)

For this algorithm to work, I recommend the data be similar in size, and similar in style / content.  For examples if we have bananas, buildings, elephants, rockets, insects of varying shapes and sizes, it would most likely just produce gibberish.

## Training the Model

After the data is loaded, let's continue with the 'tuberlin' example, you can run `python train.py --dataset_name tuberlin`

A number of flags can be set for training if you wish to experiment with the parameters.  You probably want to change these around, especially the scaling factors to better suit the sizes of your .svg data.

The default values are in `train.py`

```
--rnn_size RNN_SIZE             size of RNN hidden state
--num_layers NUM_LAYERS         number of layers in the RNN
--model MODEL                   rnn, gru, or lstm
--batch_size BATCH_SIZE         minibatch size
--seq_length SEQ_LENGTH         RNN sequence length
--num_epochs NUM_EPOCHS         number of epochs
--save_every SAVE_EVERY         save frequency
--grad_clip GRAD_CLIP           clip gradients at this value
--learning_rate LEARNING_RATE   learning rate
--decay_rate DECAY_RATE         decay rate after each epoch (adam is used)
--num_mixture NUM_MIXTURE       number of gaussian mixtures
--data_scale DATA_SCALE         factor to scale raw data down by
--keep_prob KEEP_PROB           dropout keep probability
--stroke_importance_factor F    gradient boosting of sketch-finish event
--dataset_name DATASET_NAME     name of directory containing training data
```

## Sampling a Sketch

I've included a pretrained model in `/save` so it should work out of the box.  Running `python sample.py --filename output --num_picture 10 --dataset_name kanji` will generate an .svg file containing 10 fake Kanji characters using the pretrained model.  Please run `python sample.py --help` to examine extra flags, to see how to change things like number of sketches per row, etc.

It should be straight forward to examine `sample.py` to be able to generate sketches interactively using an IPython prompt rather than in the command line.  Running `%run -i sample.py` in an IPython interactive session would generate sketches shown in the IPython interface as well as generating an .svg output.

## More useful links, pointers, datasets

- Alex Graves' [paper](http://arxiv.org/abs/1308.0850) on text sequence and handwriting generation.

- Karpathy's [char-rnn](https://github.com/karpathy/char-rnn) tool, motivation for creating sketch-rnn.

- [KanjiVG](http://kanjivg.tagaini.net/).  Fantastic Database of Kanji Stroke Order.

- Very clean TensorFlow implementation of [char-rnn](https://github.com/sherjilozair/char-rnn-tensorflow), written by [Sherjil Ozair](https://github.com/sherjilozair), where I based the skeleton of this code off of.

- [svg.path](https://pypi.python.org/pypi/svg.path).  I used this well written tool to help convert path data into line data.

- CASIA Online and Offline Chinese [Handwriting Databases](http://www.nlpr.ia.ac.cn/databases/handwriting/Download.html).  Download stroke data for written cursive Simplifed Chinese.

- How Do Humans Sketch Objects?  [TU Berlin Sketch Database](http://cybertron.cg.tu-berlin.de/eitz/projects/classifysketch/).  Would be interesting to extend this work and generate random vector art of real life stuff.

- Doraemon in [SVG format](http://yylam.blogspot.hk/2012/04/doraemon-in-svg-format-doraemonsvg.html).

- [Potrace](https://en.wikipedia.org/wiki/Potrace).  Beautiful looking tool to convert raster bitmapped drawings into SVG for potentially scaling up resolution of drawings.  Could potentially apply this to generate large amounts of training data.

- [Rendering Belzier Curve Codes](http://rosettacode.org/wiki/Bitmap/B%C3%A9zier_curves/Cubic).  I used this very useful code to convert Belzier curves into line segments.


# License

MIT
