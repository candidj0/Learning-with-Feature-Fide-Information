# Candido Ramos Joao A.

import os
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt

from mat2py import mat2py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

# ------ Settings ------
parser = argparse.ArgumentParser(description='[Pytorch] Analytical Approximation')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-e', '--epochs', default=1000, type=int, metavar='N',
                    help='number of total epochs to run (default: 1000)')
parser.add_argument('-b', '--batch-size', default=100, type=int, metavar='N',
                    help='mini-batch size (default: 100)')
parser.add_argument('-t', '--training-ratio', default=0.7, type=float, metavar='F',
                    help='Ratio for training (default: 0.7)')
parser.add_argument('-lr', '--learning-rate', default=1e-4, type=float, metavar='LR',
                    help='learning rate (default: 1e-4)')
parser.add_argument('-lg', '--lambda-reg', default=1e-2, type=float, metavar='LG',
                    help='lambda for regularizer (default: 1e-2)')
parser.add_argument('-hs', '--hidden-size', default=100, type=int, metavar='N',
                    help='hidden size (default: 100)')
parser.add_argument('-ri', '--require-improvement', default=1000, type=int, metavar='N',
                    help='if there are no improvement in require-improvement epochs stop training (default: 1000)')
parser.add_argument('-pf', '--print-freq', default=1, type=int, metavar='N',
                    help='print frequency, number of epochs between each print (default: 1)')
parser.add_argument('-c', '--cuda', default=True, type=bool, metavar='B',
                    help='use GPU (default: True)')
parser.add_argument('-p', '--pretrained', default='', type=str, metavar='PATH',
                    help='use pre-trained model (default: None)')

args = parser.parse_args()


# temps avant entrainement
t_ae = time.time()

# ------ CPU/GPU ------
dtype = torch.FloatTensor                       # CPU float type
ltype = torch.LongTensor                        # CPU long int type
if args.cuda:                                   # if args cuda is true
    if torch.cuda.is_available():               # check if cuda is available
        print('> cuda is available ! ! It\'ll run on gpu')
        dtype = torch.cuda.FloatTensor          # GPU float type
        ltype = torch.cuda.LongTensor           # GPU long int type
        cudnn.benchmark = True                  # may improve runtime
    else:
        print('> --cuda = True, but cuda is not available ! It\'ll run on cpu')

# ------ data ------
data = mat2py(args.data).convert()

if 'train' in data:
    # training set
    train_x = torch.Tensor(data['train']['x'])
    train_y = torch.Tensor(data['train']['y'])
    # validation set
    test_length = len(data['test']['x'])//2
    val_x = Variable(torch.Tensor(data['test']['x'][test_length:]), volatile=True).type(dtype)
    val_y = Variable(torch.Tensor(data['test']['y'][test_length:]), volatile=True).type(ltype)
    # test set
    test_x = Variable(torch.Tensor(data['test']['x'][:test_length])).type(dtype)
    test_y = Variable(torch.Tensor(data['test']['y'][:test_length])).type(ltype)
else :
    training_ratio = args.training_ratio             # ratio
    ts_length = int(len(data['x'])*training_ratio)
    # training set
    train_x = torch.Tensor(data['x'][:ts_length])
    train_y = torch.Tensor(data['y'][:ts_length])
    # validation set
    test_length = ((len(data['x'])-ts_length)//2) + ts_length
    val_x = Variable(torch.Tensor(data['x'][ts_length:test_length])).type(dtype)
    val_y = Variable(torch.Tensor(data['y'][ts_length:test_length])).type(ltype)
    # test set
    test_x = Variable(torch.Tensor(data['x'][test_length:])).type(dtype)
    test_y = Variable(torch.Tensor(data['y'][test_length:])).type(ltype)


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
loss_history  = []                          # to store all losses
acc_batch_history = []                      # to store all training accuracy
acc_val_history = []                        # to store all validations accuracy
best_acc_validation = 0.                    # keep best accuracy for validation set
require_improvement = args.require_improvement  # number of epochs we require an improvement
improved = 0                                # the last epoch where best_acc_validation was improved

# ------ stuff ------
'''
We need this to compute the jacobians later, because there is no function in
Pytorch that computes them directly, so we use this in backward to compute them.

for num_classes = 3, jacobian_rows_construct will be :
    jacobian_rows_construct = [[1, 0, 0]
                               [0, 1, 0]
                               [0, 0, 1]]

'''
jacobian_rows_construct = []
for c1 in range(num_classes):
    l = [0] * num_classes
    l[c1] += 1
    jacobian_rows_construct.append(l)


'''
We need this to calculate the regularizer without any loop later,
here we will keep all i, j and S[i,j] for wich S[i,j] is non-zero.

'''
ind_i = []
ind_j = []
sim = []
for i in range(input_size):
    for j in range(i+1,input_size):
        if S[i,j] != 0:
            ind_i.append(np.ones(num_classes)*i)
            ind_j.append(np.ones(num_classes)*j)
            sim.append(S[i,j])

'''
We build a matrix of length batch_size, with the i indices and another to j indices
It'll allow us to get all needed derivatives when apply in jacobians matrices later.
'''
ind_i = [ind_i for i in range(batch_size)]
ind_i = Variable(torch.Tensor(ind_i), volatile=True).type(ltype)
ind_j = [ind_j for i in range(batch_size)]
ind_j = Variable(torch.Tensor(ind_j), volatile=True).type(ltype)

# this vector will have all non-zero values in S
similarities = Variable(torch.Tensor(sim)).type(dtype)

# ------ Network ------
class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)       # first layer : input_size x hidden_size
        self.fc2 = nn.Linear(hidden_size, num_classes)      # last layer : hidden_size x num_classes

        '''
        if you want two hidden layers as the second exemple of paper
        comment self.fc3 before and uncomment bellow

        Look also forward funcion !
        '''
        #self.fc2 = nn.Linear(hidden_size, 100)              # second layer : hidden_size x 100
        #self.fc3 = nn.Linear(100, num_classes)              # last layer : 100 x num_classes


    def forward(self, x):                                   # forward step
        x = F.relu(self.fc1(x))                             # relu apply in the output of first layer
        x = self.fc2(x)                                     # last layer

        '''
        if you want two hidden layers as the second exemple of paper
        comment x = self.fc2(x) before and uncomment bellow
        '''
        #x = F.relu(self.fc2(x))                             # relu applied in the output of second layer
        #x = self.fc3(x)                                     # last layer

        return x                                            # output of model


# ------ Initialize ------
# model
net = Net(input_size, hidden_size, num_classes).type(dtype)
# loss
criterion = nn.CrossEntropyLoss().type(dtype)
# optimizer
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)


# ------ Loss + Regularizer ------
def customized_loss(pred, target, x):
    '''
    This function takes the output of the model (pred), the target values (target), and the input of the model (x)
    There are two functions:
        - the first computes all jacobians of pred w.r.t x
        - the second calculate the regularizer with the jacobians founded

    Args :
        - pred : batch_size x num_classes matrix, is the output of model for input x
        - taget : batch_size x num_classes matrix, is the target values for x
        - x : batch_size x input_size, is the current batch

    Returns :
        - the customized loss, float
    '''
    def getJacobians(pred, x):
        '''
        This function will calculate the jacobians of the model (pred) w.r.t each instance in x

        Args :
            - pred : batch_size x num_classes matrix, is the output of model for input x
            - x : batch_size x input_size, is the current batch

            Returns :
            - the jacobians : batch_size x input_size x num_classes matrix

        '''
        jacobians = Variable(torch.zeros(batch_size, input_size, num_classes)).type(dtype)
        for x_ in range(batch_size):
            for jrc in range(len(jacobian_rows_construct)):
                pred[x_].backward(torch.Tensor([jacobian_rows_construct[jrc]]).type(dtype), create_graph=True)
                jacobians[x_, :, jrc] = x.grad[x_]
                x.grad.data.zero_()
        net.zero_grad()
        return jacobians

    def regularizer(jacobians):
        '''
        This function will calculate the regularizer, only using tensor operations,
        it'll use the indices founded before and the value of similarity for each indices.

        First using gather we keep all derivatives we need, the we compute all norms,
        we multiply by our similarity vector and we sum all elements

        Args :
        - jacobians : 3D matrix, contains all jacobians for the current batch_size

        Returns :
        - the regularizer, float
        '''
        norms = torch.norm(jacobians.gather(1, ind_i)-jacobians.gather(1, ind_j),2,2)
        return (norms*similarities).sum()

    # calculate loss
    cost = criterion(pred, target)
    # calculate regularizer
    reg = regularizer(getJacobians(pred, x))

    # put them together and return
    return cost + jac.sum()

def getAccuracy(predicted, target):
    predicted = torch.max(net(predicted), 1)[1]
    correct = (predicted == target).data.sum()
    return (correct / target.data.shape[0])*100

# ------ restore ------
if args.pretrained:                                                         # if args pretrained is true
    if os.path.isfile(args.pretrained):                                     # if file exists
        print("> loading model '{}'".format(args.pretrained))               #
        checkpoint = torch.load(args.pretrained)                            # load file
        start_epoch = checkpoint['epoch']                                   # load current epoch
        num_epochs += start_epoch                                           # calculate end epoch
        net.load_state_dict(checkpoint['state_dict'])                       # restore model
        optimizer.load_state_dict(checkpoint['optimizer'])                  # restore optimizer
        losses = checkpoint['losses']                                       # restore history for loss
        print("> loaded model '{}' (epoch {})".format(args.pretrained, checkpoint['epoch']))
    else:                                                                   # no file found
        print("> no model found at '{}'".format(args.pretrained))


save_dir = 'checkpoints/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
# dispaly time before training
print('Time before training: ', time.time() - t_ae)

# ------ Training ------
print('> Start training ...')
for epoch in range(start_epoch, num_epochs):
    t_epoch = time.time()
    for i in range(num_inputs//batch_size):
        # Get data
        batch_x = train_x[i*batch_size:(i+1)*batch_size]
        batch_y = torch.max(train_y[i*batch_size:(i+1)*batch_size], 1)[1]
        # Transform data into pytorch Variables
        batch_x = Variable(batch_x.type(dtype), requires_grad=True)
        batch_y = Variable(batch_y).type(ltype)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = net(batch_x)

        # Loss
        loss = customized_loss(outputs, batch_y, batch_x)

        # Backward pass
        loss.backward()
        del loss, outputs, batch_x, batch_y
        # Optimizer step
        optimizer.step()

        loss_history.append(loss.data[0])
        batch_x.volatile = True
        acc_batch_history.append(getAccuracy(batch_x, batch_y))



        acc_val = getAccuracy(val_x, torch.max(val_y, 1)[1])
        acc_val_history.append(acc_val)

        if best_acc_validation < acc_val:
            best_acc_validation = acc_val
            state = {
                    'epoch': num_epochs,
                    'state_dict': net.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                    }
            torch.save(state, save_dir+'best_validation.pth.tar')
            improved = epoch

    if improved < epoch - require_improvement:
        print('> no improvement during last %d epochs, best accuracy validation : %.6f'%(require_improvement, best_acc_validation))
        break

    # display time for one epoch
    print('Epoch time: ', time.time() - t_epoch)

    # prints loss and curent epoch with a frequency of args.print_freq
    if (epoch+1) % args.print_freq == 0:
        print ('Epoch %d/%d, Loss: %.6f'%(epoch+1, num_epochs, loss_history[-1]))

    print(epoch)


print('\n> End !')

# restore best model
checkpoint = torch.load(save_dir+'best_validation.pth.tar')         # load best
net.load_state_dict(checkpoint['state_dict'])                       # restore model

# ------ Testing ------

acc = getAccuracy(test_x, torch.max(test_y, 1)[1])
print('Accuracy :', acc)

# ----- create logs -----

# create plots with losses, validation and mini batch accuracy
log_path = 'logs/'
if not os.path.exists(log_path):
    os.makedirs(log_path)
# loss
plt.plot(loss_history)
plt.ylabel('Loss')
plt.savefig(log_path+'loss.png')
plt.clf()
# mini-batch acc
plt.plot(acc_batch_history)
plt.ylabel('accuracy')
plt.savefig(log_path+'mini_batch_acc.png')
plt.clf()
# validation acc
plt.plot(acc_val_history)
plt.ylabel('accuracy')
plt.savefig(log_path+'validation_acc.png')
plt.clf()

# affichage du temps avant :
print('Total time: ', time.time() - t_ae)
