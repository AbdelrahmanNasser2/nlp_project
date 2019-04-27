import json
import os
import tensorflow as tf
import numpy as np

from data.dataset import get_training_data, get_test_data
from models.train_model import TrainModel
from data.utils import show_sample
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

import warnings
warnings.filterwarnings('ignore')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

tf.app.flags.DEFINE_string('train_path', os.path.abspath(os.path.join(os.path.dirname( "__file__" ), '..', 'data/tf_records/trainingg_file_0.tfrecord')),
                           'Path for the training data.')
tf.app.flags.DEFINE_string('test_path', os.path.abspath(os.path.join(os.path.dirname("__file__"), '..', 'data/tf_records/testt_file_0.tfrecord')),
                           'Path for the test data.')

tf.app.flags.DEFINE_string('word2idx', os.path.abspath(os.path.join(os.path.dirname("__file__"), '..', 'data/preprocessed/word2idx.txt')), 
                           'Path for the word2idx dictionary.')

tf.app.flags.DEFINE_string('checkpoints_path', os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'checkpoints/model.ckpt')), 
                           'Path for the test data.')

tf.app.flags.DEFINE_integer('num_epoch', 10,
                            'Number of training epoch.'
                            )
tf.app.flags.DEFINE_integer('batch_size', 32,
                            'Batch size of the training set.'
                            )
tf.app.flags.DEFINE_float('learning_rate', 0.0015,
                          'Learning rate of optimizer.'
                          )

tf.app.flags.DEFINE_string('architecture', 'bidirectional',
                          'Type of LSTM-Architecture, choose between "unidirectional" or "bidirectional"'
                          )

tf.app.flags.DEFINE_integer('lstm_units', 100,
                            'Number of the LSTM hidden units.'
                            )

tf.flags.DEFINE_float('dropout_keep_prob', 0.5,
                      '0<dropout_keep_prob<=1. Dropout keep-probability')

tf.app.flags.DEFINE_integer('embedding_size', 100,
                            'Dimension of the embedding vector for the vocabulary.'
                            )
tf.app.flags.DEFINE_integer('num_classes', 2,
                            'Number of output classes.'
                            )

tf.app.flags.DEFINE_integer('n_train_samples', 1000,
                            'Number of all training sentences.'
                            )

tf.app.flags.DEFINE_integer('n_test_samples', 1000,
                            'Number of all training sentences.'
                            )
tf.app.flags.DEFINE_float('required_acc_checkpoint', 0.7,
                          'The accuracy on the test set that must be achieved, before any checkpoints are saved.'
                          )


FLAGS = tf.app.flags.FLAGS



def main(_):


    with open(FLAGS.word2idx) as json_file:  
        word2idx = json.load(json_file)

    training_graph=tf.Graph()
    
    with training_graph.as_default():
        
        train_model=TrainModel(FLAGS, len(word2idx))
        
        training_dataset = get_training_data(FLAGS)
        test_dataset = get_test_data(FLAGS)
        
        iterator_train = training_dataset.make_initializable_iterator()
        iterator_test = test_dataset.make_initializable_iterator()
        
        x_train, y_train, _, seq_length_train = iterator_train.get_next()
        x_test, y_test, _, seq_length_test =iterator_test.get_next()


        dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
        
        logits, probs=train_model.compute_prediction(x_train, seq_length_train, dropout_keep_prob, reuse_scope=False)
        loss=train_model.compute_loss(logits, y_train)
        train_op=train_model.train(loss)
        accuracy_train = train_model.compute_accuracy(probs, y_train)
        
        x=tf.identity(x_test)
        logits_test, probs_test=train_model.compute_prediction(x_test, seq_length_test, dropout_keep_prob, reuse_scope=True) 
        accuracy_test = train_model.compute_accuracy(probs_test, y_test)

        # print("prec",y_test)
        saver=tf.train.Saver()

    with tf.Session(graph=training_graph) as sess:

        # print(sess.run(type))
        sess.run(tf.global_variables_initializer())


        n_batches=int(FLAGS.n_train_samples/FLAGS.batch_size)   

        for epoch in range(FLAGS.num_epoch):
            
            sess.run(iterator_train.initializer)
            sess.run(iterator_test.initializer)
            
            traininig_loss=0
            training_acc=0



            feed_dict={dropout_keep_prob:0.5}

            prb_test = sess.run(probs_test, feed_dict)
            y_ttest = sess.run(y_test, feed_dict)

            prb_test2 = []

            y_ttest2 = []

            for x in prb_test:
                if x[0] >= 0.5:
                    prb_test2.append(1)
                else:
                    prb_test2.append(0)


            for y in y_ttest:
               y_ttest2.append(y[0])



            # print(type(y_ttest))
        

            for n_batch in range(0, n_batches):
              
                _, l, acc, logits_, probs_=sess.run((train_op, loss, accuracy_train, logits, probs), feed_dict)

                traininig_loss+=l
                training_acc+=acc
                

                  
            feed_dict={dropout_keep_prob:1.0}
                
            acc_avg_test=sess.run(accuracy_test, feed_dict)
            acc_avg_train=training_acc/n_batches




            print('Epoch: %i, Accuracy: %.3f'%(epoch, acc_avg_test))

            print("\tPrecision: %1.3f" % precision_score(y_ttest2, prb_test2))
            print("\tRecall: %1.3f" % recall_score(y_ttest2, prb_test2))
            print("\tF1: %1.3f\n" % f1_score(y_ttest2, prb_test2))




if __name__ == "__main__":
    
    tf.app.run() 

