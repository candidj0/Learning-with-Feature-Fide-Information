import h5py
from scipy import sparse as ssp
import numpy as np
import random

class mat2py:
    def __init__(self, filename, train=True, test=True, X=True, Y=True, S=True, S10=False, Voc=False):
        self.f = h5py.File(filename)
        self.X = X
        self.Y = Y
        self.train = train
        self.test = test
        self.S = S
        self.S10 = S10
        self.Voc = Voc

        self.data = {}

    def convert(self):
        ''' Convert data from matlab type to python type '''
        if self.f.__contains__('x'):
            if self.X:
                self.data['x'] = self.getSparseMatrix('x')
            if self.Y:
                self.data['y'] = self.getSparseMatrix('y')
            self.shuffle()
        if self.f.__contains__('train'):
            if self.train:
                self.data['train'] = {}
                self.data['train']['x'] = self.getSparseMatrix('x','train')
                self.data['train']['y'] = self.getSparseMatrix('y','train')
            if self.test:
                self.data['test'] = {}
                self.data['test']['x'] = self.getSparseMatrix('x','test')
                self.data['test']['y'] = self.getSparseMatrix('y','test')
        if self.S:
            self.data['s'] = np.array(self.f['SS'])
        if self.S10:
            self.data['s10'] = self.getSparseMatrix('S10')
        if self.Voc:
            self.data['voc'] = np.array(self.f['vocabulary'])
        return self.data

    def getSparseMatrix(self, name, ttype=None):
        ''' Convert sparse matrix into array '''
        if self.f.__contains__('x'):
            data = self.f[name]['data']
            ir = self.f[name]['ir']
            jc = self.f[name]['jc']
        if self.f.__contains__('train'):
            data = self.f[ttype][name]['data']
            ir = self.f[ttype][name]['ir']
            jc = self.f[ttype][name]['jc']

        return ssp.csc_matrix((data, ir, jc)).toarray()

    def shuffle(self):
        ''' shuffle the data if there are no train and test set '''
        s = []
        for i in range(0, len(self.data['x'])):
            s.append([self.data['x'][i], self.data['y'][i]])
        random.shuffle(s)
        s = np.array(s)
        self.data['x'] = list(s[:,0])
        self.data['y'] = list(s[:,1])
