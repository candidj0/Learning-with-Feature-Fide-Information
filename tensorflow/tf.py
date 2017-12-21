# Candido Ramos Joao A.

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import time
import argparse
import numpy as np

from mat2py import mat2py

import tensorflow as tf
from tensorflow.python.client import device_lib

# ------ Settings ------
parser = argparse.ArgumentParser(description='[TensorFlow] Analytical Approximation')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-e', '--epochs', default=1000, type=int, metavar='N',
                    help='number of total epochs to run (default: 1000)')
parser.add_argument('-b', '--batch-size', default=100, type=int, metavar='N',
                    help='mini-batch size (default: 100)')
parser.add_argument('-t', '--training-ratio', default=0.7, type=float, metavar='F',
                    help='Ratio for training (default: 0.7)')
parser.add_argument('-lr', '--learning-rate', default=1e-2, type=float, metavar='LR',
                    help='learning rate (default: 1e-2)')
parser.add_argument('-lg', '--lambda-reg', default=1e-4, type=float, metavar='LG',
                    help='lambda for regularizer (default: 1e-4)')
parser.add_argument('-hs', '--hidden-size', default=100, type=int, metavar='N',
                    help='hidden size (default: 100)')
parser.add_argument('-ri', '--require-improvement', default=1000, type=int, metavar='N',
                    help='if there are no improvement in require-improvement epochs stop training (default: 1000)')
parser.add_argument('-pf', '--print-freq', default=1, type=int, metavar='N',
                    help='print frequency, number of epochs between each print (default: 1)')
parser.add_argument('-c', '--cuda', default=True, type=bool, metavar='B',
                    help='use GPU (default: True)')

args = parser.parse_args()

# temps avant entrainement
t_ae = time.time()


# ------ CPU/GPU ------
def get_available_gpus():
    ''' This function check if there are GPU's and return their names'''
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

# default device CPU
device_name = "/cpu:0"

if args.cuda:                                                       # if args cuda is true
    gpus = get_available_gpus()                                     # check if cuda is available
    if len(gpus):
        device_name = gpus[0]                                       # put device name to /gpu:0
        print('> cuda is available ! It\'ll run on gpu')
    else:
        print('> --cuda = True, but cuda is not available ! It\'ll run on cpu')

# ------ data ------
data = mat2py(args.data).convert()

if 'train' in data:
    # training set
    train_x = data['train']['x']
    train_y = data['train']['y']
    # validation set
    test_length = len(data['test']['x'])//2
    val_x = data['test']['x'][test_length:]
    val_y = data['test']['y'][test_length:]
    # test set
    test_x = data['test']['x'][:test_length]
    test_y = data['test']['y'][:test_length]
else :
    training_ratio = args.training_ratio             # ratio
    ts_length = int(len(data['x'])*training_ratio)
    # training set
    train_x = data['x'][:ts_length]
    train_y = data['y'][:ts_length]

    test_length = ((len(data['x'])-ts_length)//2) + ts_length
    val_x = data['x'][ts_length:test_length]
    val_y = data['y'][ts_length:test_length]
    # test set
    test_x = data['x'][test_length:]
    test_y = data['y'][test_length:]


# similarity matrix
S = data['s']

# info of data
num_inputs = len(train_x)                   # number of instances
input_size = len(train_x[0])                # number of features per instance
num_classes = len(train_y[0])               # number of classes


# ------ parameters ------
hidden_size = args.hidden_size              # number of neurones in hidden layer
start_epoch = 0                             # first epoch (usefull when we store a NN and want continue training)
num_epochs = args.epochs                    # number of epochs
batch_size = args.batch_size                # length of the batch_size
learning_rate = args.learning_rate          # learning rate (used in adam)
best_acc_validation = 0.                    # keep best accuracy for validation set
require_improvement = args.require_improvement  # number of epochs we require an improvement
improved = 0                                # the last epoch where best_acc_validation was improved

# ------ Network ------
def Net(x, weights, biases):
    with tf.name_scope("model"):
        h_layer_1 = tf.add(tf.matmul(x, weights['h_l1']), biases['b1'])
        h_layer_1 = tf.nn.relu(h_layer_1)
        out_layer = tf.add(tf.matmul(h_layer_1, weights['out_l']), biases['out'])
    return out_layer

# ------ Jacobians ------
def jacobian(y, x, i):
    '''
    This funcion computes the jacobian of y w.r.t x

    Args :
        - y : 1 x num_classes, the prediction of the model for input x
        - x : 1 x input_size, an instance (input)

    Return :
        - the jacobian of y w.r.t. x : num_classes x input_size
    '''
    n = tf.shape(y)[0]
    loop_vars = [
        tf.constant(0, tf.int32),
        tf.TensorArray(tf.float32, size=n),
    ]
    _, jac = tf.while_loop(
        lambda j, _: j < n,
        lambda j, result: (j+1, result.write(j, tf.gradients(y[j], x)[0][i])),
        loop_vars)
    return jac.stack()

def tf_jacobians(y, x, n):
    '''
    This funcion calls jacobian() for each instance, and return all jacobians.

    Args :
        - y : batch_size x num_classes, the predictions of the model for inputs x
        - x : batch_size x input_size, mini batch

    Return :
        - all jacobians of y w.r.t. x : batch_size x num_classes x input_size
    '''
    loop_vars = [
        tf.constant(0, tf.int32),
        tf.TensorArray(tf.float32, size=n),
    ]
    _, list_jacobians = tf.while_loop(
        lambda i, _: i < n,
        lambda i, result: (i+1, result.write(i, jacobian(y[i], x, i))),
        loop_vars)
    return list_jacobians.stack()


# ------ stuff ------
'''
We need this to calculate the regularizer without any loop later,
here we will keep all i, j and S[i,j] for wich S[i,j] is non-zero.

'''
ind_i = []
ind_j = []
similarities = []
for i in range(input_size):
    for j in range(i+1,input_size):
        if S[i,j] != 0:
            ind_i.append(i)
            ind_j.append(j)
            similarities.append(S[i,j])

# start graph
with tf.device(device_name):
    # define placeholders
    with tf.name_scope('input'):
        x = tf.placeholder("float", [None, input_size], name='x')
        y = tf.placeholder("float", [None, num_classes], name='y')

    # define weights
    with tf.name_scope("weights"):
        weights = {
            'h_l1': tf.Variable(tf.random_normal([input_size, hidden_size])),
            'out_l': tf.Variable(tf.random_normal([hidden_size, num_classes]))
        }
    # define biases
    with tf.name_scope("biases"):
        biases = {
            'b1': tf.Variable(tf.random_normal([hidden_size])),
            'out': tf.Variable(tf.random_normal([num_classes]))
        }

    # Initialize model
    net = Net(x, weights, biases)

    # ------ Loss + Regularizer ------
    with tf.name_scope("Loss"):
        # define loss
        with tf.name_scope("cross_entropy"):
            ce_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=net, labels=y))
        # define regularizer
        with tf.name_scope("regularizer"):
            with tf.name_scope("jacobians"):
                jacobians = tf_jacobian(net, x, batch_size)
            with tf.name_scope("regularizer_cal"):
                regularizer = tf.reduce_sum(tf.norm(tf.gather(jacobians, ind_i, axis=2) - tf.gather(jacobians, ind_j, axis=2), axis=1)*similarities)
        # get final loss by adding loss and regularizer
        customized_loss = tf.add(ce_loss, args.lambda_reg*regularizer)

    # define optimizer
    with tf.name_scope("optimizer"):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(customized_loss)

    # define accuracy
    with tf.name_scope("accuracy"):
        prediction = tf.nn.softmax(net)
        correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # create a summary for our cost
    tf.summary.scalar("cost", customized_loss)
    # create a summary for our training accuracy
    tf.summary.scalar("accuracy_train", accuracy)
    # merge all summaries
    summary_op = tf.summary.merge_all()

    # create anothe summary of accuracy, but this time for validation set
    acc_val = tf.summary.scalar("accuracy_validation", accuracy)

    # create repo for logs if doesn't exit
    log_path = './logs'
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    # create log writer, and save graphe
    writer = tf.summary.FileWriter(log_path, graph=tf.get_default_graph())

    # create saver to save model
    saver = tf.train.Saver()
    # create repo to save model if doesn't exit
    save_dir = 'checkpoints/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, 'best_validation')

    # some configurations
    config = tf.ConfigProto(allow_soft_placement = True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    #  display time before training
    print('Time before training: ', time.time() - t_ae)

    # ------ Training ------
    print('Start training ...')
    # Initialize all Variables
    sess.run(tf.global_variables_initializer())
    for epoch in range(start_epoch, num_epochs):
        t_epoch = time.time()
        for i in range(num_inputs//batch_size):
            # Get data
            batch_x = train_x[i*batch_size:(i+1)*batch_size]
            batch_y = train_y[i*batch_size:(i+1)*batch_size]

            # optimize, get loss, training accuracy
            _, loss, summary, acc = sess.run([optimizer, customized_loss, summary_op, accuracy], feed_dict = {x:batch_x, y:batch_y})
            # get summary for validation accuracy, and validation accuracy
            val_summary_str, val = sess.run([acc_val, accuracy], feed_dict={x:val_x, y:val_y})

            # if global best_acc_validation is smallest the new validation accuracy val
            if best_acc_validation < val:
                best_acc_validation = val                           # update
                saver.save(sess=sess, save_path=save_path)          # save model
                improved = epoch                                    # update

            # write summaries
            writer.add_summary(val_summary_str, epoch * (num_inputs//batch_size) + i)
            writer.add_summary(summary, epoch * (num_inputs//batch_size) + i)

        # if there is no improvement of validation accuracy in last require_improvement epochs stops training
        if improved < epoch - require_improvement:
            print('> no improvement during last %d epochs, best accuracy validation : %.6f'%(require_improvement, best_acc_validation))
            break

        # display time for one epoch
        print('Epoch time: ', time.time() - t_epoch)

        # prints loss and curent epoch with a frequency of args.print_freq
        if (epoch+1) % args.print_freq == 0:
            print ('Epoch %d/%d, Loss: %.6f'%(epoch+1, num_epochs, loss))

    print('End !')

    # restore best model
    saver.restore(sess=sess, save_path=save_path)
    # ------ Testing ------
    acc = sess.run(accuracy, {x:test_x, y:test_y})*100
    print("Accuracy:", acc)

    # display total time
    print('Total time: ', time.time() - t_ae)
