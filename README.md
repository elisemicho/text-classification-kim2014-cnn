# CharCNN

Python 3.5, Tensorflow 1.4.0 Implementation of word-level CNN for Arabic Dialect Identification

## CNN

* Inspirations
Wildml tutorial: Implementing a CNN for Text Classification in TensorFlow: http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/
https://github.com/dennybritz/cnn-text-classification-tf 

Stanford classes: Tensorflow for Deep Learning Research: https://web.stanford.edu/class/cs20si/syllabus.html
https://github.com/chiphuyen/stanford-tensorflow-tutorials/tree/master/examples

* Model

![Alt text](CNNmodel.png?raw=true "CNN model")

	# Data loading parameters
	self.data_name = sys.argv[1]
	self.data_folder = "../data/"+self.data_name
	self.mode = sys.argv[2]
	self.data_files = ["EGY", "GLF", "LAV", "MSA", "NOR"]
	self.train_file = "train.words"
	self.dev_file = "dev.words"
	self.batch_size = 64

	# Model Hyperparameters
	# Embeddings
	self.vocab_size = 50000
	self.embed_size = 128

	# Convolutional layers
	self.filter_sizes = [3,4,5]
	self.num_filters = 3
	self.dropout_keep_prob = 0.5
	self.l2_reg_lambda = 0.0
	self.l2_loss = 0.0

	# Training
	self.lr = 0.5
	self.training=False
	self.n_epochs = 200
	self.skip_step = 10

* Jupyter notebooks available to display the shape of layers and debug

## Adapting the model to VarDial data

* Data
	VarDial 2017: https://github.com/qcri/dialectID/tree/master/data
	VarDial 2018: http://alt.qcri.org/vardial2018/index.php?id=campaign

	data
	├── vardial2017
	│   ├── EGY
	│   ├── GLF
	│   ├── LAV
	│   ├── MSA
	│   └── NOR
	├── vardial2017-sample
	│   ├── EGY
	│   ├── GLF
	│   ├── LAV
	│   ├── MSA
	│   └── NOR
	├── vardial2018
	│   ├── dev.words
	│   └── train.words
	├── vardial2018-bigsample
	│   ├── dev.words
	│   └── train.words
	└── vardial2018-sample
	    ├── dev.words
	    └── train.words

* How to run to train and eval on dev set at the end of each epoch
```sh
$ python char_cnn.py <data_folder> <mode>
```

* How to set arguments
```
<data_folder> = {vardial2017, vardial2017-sample}, <mode> = "many"
<data_folder> = {vardial2018, vardial2018-sample, vardial2018-bigsample}, <mode> = "other"
```

## Current problems

* Accuracy stays null or very low all training long