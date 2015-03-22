'''
single layer denoising autoencoder (dA.py) trained via log loss from the output of MLP.
'''
import os
import sys
import time
import pdb
import cPickle as pickle
import numpy

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from logistic_sgd import load_data
from utils import tile_raster_images

try:
    import PIL.Image as Image
except ImportError:
    import Image

from dA import *
from mlp import *

class semantic_dA(object):
    def __init__(
        self,
        rng,
        theano_rng,
        input,
        n_in,
        n_hidden,
        n_out,
        f_load_MLP,
        f_load_DA=None,
    ):
        self.x = input
        self.da = dA(                
            numpy_rng=rng,
            theano_rng=theano_rng,
            input=self.x,
            n_visible=28 * 28,
            n_hidden=n_hidden,
            f_load=f_load_DA)
        self.mlp = MLP(
            rng=rng,
            input=self.da.output,
            f_load = f_load_MLP
        )

        self.params = self.da.params
        self.output = self.da.output
        self.L1 = abs(self.da.W).sum()
        self.L2 = (self.da.W ** 2).sum()
    
    def get_cost_updates(self, learning_rate, y_true):
        cost = self.mlp.negative_log_likelihood(y_true)
        gparams = T.grad(cost, self.params)
        updates = [
            (param, param - learning_rate*gparam)
            for param, gparam in zip(self.params, gparams)
            ]
        return (cost, updates)
    
    def dump(self, f_dump):
        self.da.dump(f_dump)


def train_da_semantic(sda, train_set_x, train_set_y, corruption=0):
    #trains a denoising autoencoder from MNIST, then returns the trained dA object.
    learning_rate=0.1
    training_epochs=15
    batch_size=20

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size

    # allocate symbolic variables for the data
    index = T.lscalar()    # index to a [mini]batch
    y = T.ivector('y') #labels    
    
    cost, updates = sda.get_cost_updates(
        learning_rate=learning_rate,
        y_true=y)
    
    train_sda = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            sda.x: train_set_x[index*batch_size: (index+1)*batch_size],
            y: train_set_y[index*batch_size: (index+1)*batch_size]
            }
        )
    
    start_time = time.clock()
    
    #TRAINING
    for epoch in xrange(training_epochs):
        c = []
        for batch_index in xrange(n_train_batches):
            c.append(train_sda(batch_index))

        print 'Training epoch %d, cost ' % epoch, numpy.mean(c)

    end_time = time.clock()
    training_time = (end_time - start_time)
    print >> sys.stderr, ('The 50 neuron code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((training_time) / 60.))

    return sda

def train_da_semantic_elaborate(sda, datasets,
                                learning_rate=0.01,
                                L1_reg=0.00,
                                L2_reg=0.0001,
                                n_epochs=1000,
                                batch_size=20):
                                
    #load data
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    #cost is the sum of neg log likelihood of MLP on output of DA + regularization
    cost = (sda.mlp.negative_log_likelihood(y) + L1_reg * sda.L1 + L2_reg * sda.L2)
    
    test_model = theano.function(
        inputs=[index],
        outputs=sda.mlp.errors(y),
        givens={
            sda.x: test_set_x[index * batch_size:(index + 1) * batch_size],
            y: test_set_y[index * batch_size:(index + 1) * batch_size]
            }
        )
    
    validate_model = theano.function(
        inputs=[index],
        outputs=sda.mlp.errors(y),
        givens={
            sda.x: valid_set_x[index * batch_size:(index + 1) * batch_size],
            y: valid_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )
    
    #gradient/updates
    gparams = [T.grad(cost, param) for param in sda.params]
    updates = [ (param, param-learning_rate*gparam)
                for param,gparam in zip(sda.params,gparams)]
                
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            sda.x: train_set_x[index*batch_size: (index+1)*batch_size],
            y: train_set_y[index*batch_size: (index+1)*batch_size]
            }
        )
    
    #TRAINING
    print '... training'
    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = time.clock()

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):
            minibatch_avg_cost = train_model(minibatch_index)
            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in xrange(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)

                print(
                    'epoch %i, minibatch %i/%i, validation error %f %%' %
                    (
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        this_validation_loss * 100.
                    )
                )

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    #improve patience if loss improvement is good enough
                    if (
                        this_validation_loss < best_validation_loss *
                        improvement_threshold
                    ):
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = [test_model(i) for i
                                   in xrange(n_test_batches)]
                    test_score = numpy.mean(test_losses)

                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

            if patience <= iter:
                done_looping = True
                break

    end_time = time.clock()
    print(('Optimization complete. Best validation score of %f %% '
           'obtained at iteration %i, with test performance %f %%') %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))    
    return sda


def train_autos_semantic():
    n_hiddens = [10,25,50,100]
    corruption = 0.2
    #load data
    dataset='mnist.pkl.gz'
    datasets = load_data(dataset)
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]


    #parameters for sDAs
    x = T.matrix('x')
    y = T.ivector('y')
    rng = numpy.random.RandomState(1234)
    theano_rng = RandomStreams(rng.randint(2 ** 30))

    #initialize sDAs
    sDAs_randinit = []
    sDAs_l2init = []
    sDAs_corrinit = []
    for n_hidden in n_hiddens:
        print 'defining sdA for n_hidden = ',n_hidden
        sDAs_l2init.append(semantic_dA(
                rng=rng,
                theano_rng=theano_rng,
                input=x,
                n_in=28 * 28,
                n_hidden=n_hidden,
                n_out=10,
                f_load_MLP = open('../data/MLP_params.p','r'),
                f_load_DA = open('../data/dA_l2/dA_l2_nhid'+str(n_hidden)+'.p','r')
                ))
        sDAs_corrinit.append(semantic_dA(
                rng=rng,
                theano_rng=theano_rng,
                input=x,
                n_in=28 * 28,
                n_hidden=n_hidden,
                n_out=10,
                f_load_MLP = open('../data/MLP_params.p','r'),
                f_load_DA = open('../data/dA_l2/dA_l2_nhid'+str(n_hidden)+'_corr'+str(corruption)+'.p','r')
                ))
        sDAs_randinit.append(semantic_dA(
                rng=rng,
                theano_rng=theano_rng,
                input=x,
                n_in=28 * 28,
                n_hidden=n_hidden,
                n_out=10,
                f_load_MLP = open('../data/MLP_params.p','r')
                ))


        #paths to dumped versions
        prefix = 'retry_'
        dir = '../data/sdA_l2init/'
        path_l2init = dir+ prefix+ 'sDA_l2init_elaborate_nhid'+str(n_hidden)+'.p'
        path_randinit = dir+prefix+'sDA_randinit_elaborate_nhid'+str(n_hidden)+'.p'
        path_corrinit = dir+prefix+'retry_sDA_corrinit'+str(corruption)+'_elaborate_nhid'+str(n_hidden)+'.p'
        #paths to pictures
        picdir = dir+'pics/'
        path_pic_l2init = picdir+prefix+'sdA_l2init_nhid'+str(n_hidden)+'.png'
        path_pic_randinit = picdir+prefix+'sdA_randinit_nhid'+str(n_hidden)+'.png'
        path_pic_corrinit = picdir+prefix+'sdA_corrinit_nhid'+str(n_hidden)+'_corr'+str(corruption)+'.png'
        
        #train and test
        #first for l2init
        if not os.path.isfile(path_l2init):
            print 'training l2_init for n_hidden = ',n_hidden
            train_da_semantic_elaborate(sDAs_l2init[-1], datasets)
            print 'storing dA to file'
            sDAs_l2init[-1].dump(open(path_l2init,'w'))
        else: 
            sDAs_l2init[-1].da.load(open(path_l2init,'r'))
        
        print 'generating image'
        reconstruct = theano.function(
            [sDAs_l2init[-1].x],
            sDAs_l2init[-1].output
            )
        image = Image.fromarray(tile_raster_images(X=reconstruct(valid_set_x.get_value()),img_shape=(28, 28), tile_shape=(10, 10),tile_spacing=(1, 1)))
        image.save(path_pic_l2init)
        
        #then corruption initialized corrinit
        if not os.path.isfile(path_corrinit):
            print 'training corr_init for n_hidden = ',n_hidden,' and corruption ',corruption
            train_da_semantic_elaborate(sDAs_corrinit[-1], datasets)
            print 'storing dA to file'
            sDAs_corrinit[-1].dump(open(path_corrinit,'w'))
        else: 
            sDAs_corrinit[-1].da.load(open(path_corrinit,'r'))
        
        print 'generating image'
        reconstruct = theano.function(
            [sDAs_corrinit[-1].x],
            sDAs_corrinit[-1].output
            )
        image = Image.fromarray(tile_raster_images(X=reconstruct(valid_set_x.get_value()),img_shape=(28, 28), tile_shape=(10, 10),tile_spacing=(1, 1)))
        image.save(path_pic_corrinit)


        #same for randinit
        if not os.path.isfile(path_randinit):
            print 'training rand_init for n_hidden = ',n_hidden
            train_da_semantic_elaborate(sDAs_randinit[-1],datasets)
            print 'storing dA to file'
            sDAs_randinit[-1].dump(open(path_randinit,'w'))
        else:
            sDAs_randinit[-1].da.load(open(path_randinit,'r'))
        
        print 'generating image'
        reconstruct = theano.function(
            [sDAs_randinit[-1].x],
            sDAs_randinit[-1].output
            )
        image = Image.fromarray(tile_raster_images(X=reconstruct(test_set_x.get_value()),img_shape=(28, 28), tile_shape=(10, 10),tile_spacing=(1, 1)))
        image.save(path_pic_randinit)
            

if __name__ == '__main__':
    train_autos_semantic()
