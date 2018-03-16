import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import time 
import sys

import numpy as np
import tensorflow as tf

import utils


class CharCNNModel:
    """ Build the graph """
    def __init__(self):

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
        self.global_step = tf.get_variable('global_step', initializer=tf.constant(0), trainable=False)

    def _import_data(self):
        """ Step 1: import data
        """
        with tf.variable_scope('data'):
            # Read in train and dev data (sentences as word ids and padded, labels as one-hot vectors)
            #train_data, test_data = utils.get_vardial_dataset_from_many_files(self.batch_size, self.data_folder, self.data_files)
            if(self.mode == 'many'):
                x_train, x_dev, y_train, y_dev = utils.get_vardial_dataset_from_many_files(self.data_folder, self.data_files)
            else:
                x_train, x_dev, y_train, y_dev = utils.get_vardial_dataset_from_train_dev_file(self.data_folder, self.train_file, self.dev_file)

            #Create datasets
            train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
            train_data = train_data.shuffle(10000) # to shuffle your data
            train_data = train_data.batch(self.batch_size)

            test_data = tf.data.Dataset.from_tensor_slices((x_dev, y_dev))
            test_data = test_data.batch(self.batch_size)

            # Create iterators
            iterator = tf.data.Iterator.from_structure(train_data.output_types, 
                                                   train_data.output_shapes)
            self.sentence, self.label = iterator.get_next()
            # shape = [batch_size, sentence_length],[batch_size, num_classes]

            self.sentence_length = self.sentence.shape[1].value
            self.num_classes = self.label.shape[1].value
            self.n_test = len(y_dev)

            self.train_init = iterator.make_initializer(train_data)  # initializer for train_data
            self.test_init = iterator.make_initializer(test_data)    # initializer for test_data

    def _create_embedding(self):
        """ Step 2 + 3: define weights and embedding lookup
        """
        with tf.variable_scope('word_embedding'):
            self.embed_matrix = tf.get_variable('embed_matrix', 
                                                shape=[self.vocab_size, self.embed_size],
                                                initializer=tf.random_uniform_initializer())
            # shape = [vocab_size, embed_size]
            
            self.embedded_words = tf.nn.embedding_lookup(self.embed_matrix, self.sentence, name='embedding')
            # shape = [batch_size, sentence_length, embed_size]

            self.embedded_words_expanded = tf.expand_dims(self.embedded_words, -1)
            # shape = [batch_size, sentence_length, embed_size, 1]

    def _create_conv_maxpool(self):
        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(self.filter_sizes):
            with tf.variable_scope("conv-maxpool-%d" % filter_size):

                # Convolution layer
                conv = tf.layers.conv2d(inputs=self.embedded_words_expanded,
                                  filters=self.num_filters,
                                  kernel_size=[filter_size, self.embed_size],
                                  padding='SAME',
                                  activation=tf.nn.relu,
                                  name='conv')
                # shape = [batch_size, sequence_length, embed_size, num_filters]

                # Maxpooling over the outputs
                pooled = tf.layers.max_pooling2d(inputs=conv, 
                                        pool_size=(self.sentence_length - filter_size + 1, 1), 
                                        strides=[filter_size, self.embed_size],
                                        name='pool')
                # shape =  [batch_size, 1, 1, num_filters]

                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = self.num_filters * len(self.filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        # shape = [batch_size, 1, 1, num_filters_total]
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])
        # shape = [batch_size, num_filters_total]

        # Add dropout
        with tf.variable_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)
            # shape = [batch_size, num_filters_total]

        with tf.variable_scope("fully-connected"):
            self.scores = tf.layers.dense(self.h_drop, self.num_classes,
                        kernel_initializer=tf.contrib.layers.xavier_initializer())
            # shape = [batch_size, num_classes]
            self.predictions = tf.argmax(self.scores, 1, name="predictions")
            # shape = [batch_size, 1]

    def _create_loss(self):
        """ Step 4: define the loss function """
        with tf.variable_scope('loss'):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.label)
            # shape = [batch_size, 1]

            self.loss = tf.reduce_mean(losses, name="loss") + self.l2_reg_lambda * self.l2_loss

        with tf.variable_scope('accuracy'):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.label, 1))
            # shape = [batch_size, 1]

            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

    def _create_optimizer(self):
        """ Step 5: define optimizer """
        with tf.variable_scope('optimizer'):
            self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss, 
                                                              global_step=self.global_step)

    def _create_summaries(self):
        with tf.variable_scope('summaries'):
            tf.summary.scalar('loss', self.loss)
            tf.summary.scalar('accuracy', self.accuracy)
            tf.summary.histogram('histogram_loss', self.loss)
            self.summary_op = tf.summary.merge_all()

    def build_graph(self):
        """ Build the graph for our model """
        self._import_data()
        self._create_embedding()
        self._create_conv_maxpool()
        self._create_loss()
        self._create_optimizer()
        self._create_summaries()

    """ Phase 2: Train the graph """

    def train_one_epoch(self, sess, saver, init, writer, epoch, step):
        start_time = time.time()
        sess.run(init) 
        self.training = True
        total_loss = 0
        n_batches = 0
        try:
            while True:
                _, l, acc, summaries = sess.run([self.optimizer, self.loss, self.accuracy, self.summary_op])
                writer.add_summary(summaries, global_step=step)
                if (step + 1) % self.skip_step == 0:
                    print('step {0}, loss {1}'.format(step, l))
                step += 1
                total_loss += l
                n_batches += 1
        except tf.errors.OutOfRangeError:
            pass
        saver.save(sess, 'checkpoints/'+self.data_name, step)
        print('Average loss {1} for epoch {0}, took {2} sec'.format(epoch+1, total_loss/n_batches, time.time() - start_time))
        return step

    def eval_once(self, sess, init, writer, epoch, step):
        start_time = time.time()
        sess.run(init)
        self.training = False
        total_correct_preds = 0
        try:
            while True:
                accuracy_batch, summaries = sess.run([self.accuracy, self.summary_op])
                writer.add_summary(summaries, global_step=step)
                total_correct_preds += accuracy_batch
        except tf.errors.OutOfRangeError:
            pass

        print('Accuracy {1} for epoch {0}, took {2} sec'.format(epoch+1, total_correct_preds/self.n_test, time.time() - start_time))

    def train(self):
        '''
        The train function alternates between training one epoch and evaluating
        '''
        utils.safe_mkdir('checkpoints')
        writer = tf.summary.FileWriter('./graphs/'+self.data_name, tf.get_default_graph())

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/'+self.data_name))
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
            
            step = self.global_step.eval()

            for epoch in range(self.n_epochs):
                print('Epoch {0}'.format(epoch+1))
                step = self.train_one_epoch(sess, saver, self.train_init, writer, epoch, step)
                self.eval_once(sess, self.test_init, writer, epoch, step)
        writer.close()


def main():

    model = CharCNNModel()
    model.build_graph()
    model.train()

if __name__ == '__main__':
    main()