"""
Stacked semantic DAs. Constructed from stacked DA code.
"""
import os
import sys
import time
import pdb
import pickle

import numpy

import theano
theano.config.exception_verbosity='high'
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from logistic_sgd import LogisticRegression, load_data
from mlp import HiddenLayer, MLP, HiddenLayer_ReLU
from dA import dA
from stacked_da import SdA


# start-snippet-1
class ssDA(object):
    """Semantic Stacked denoising auto-encoder class (SdA)

    A stacked denoising autoencoder model is obtained by stacking several
    dAs. The hidden layer of the dA at layer `i` becomes the input of
    the dA at layer `i+1`. The first layer dA gets as input the input of
    the SdA, and the hidden layer of the last dA represents the output.
    Note that after pretraining, the SdA is dealt with as a normal MLP,
    the dAs are only used to initialize the weights.
    
    Semantic distortion comes from an additional MLP layer tacked on after reconstruction by the autoencoding chain.
    """
    def dump(self,f_dump):
        #f_dump: file object
        #        dump parameters in the form of a dictionary to this file.
        #params of relevance: sigmoid and out_sigmoid's W and b.
        # also, n_in and n_out for each.
        hidden_paramdicts = []
        for layer in (self.sigmoid_layers + self.out_sigmoid_layers):
            hidden_paramdicts.append({'W': layer.W.get_value(),
                                  'b': layer.b.get_value(),
                                  'n_in': layer.n_in,
                                  'n_out': layer.n_out})
        
        pickle.dump(hidden_paramdicts,f_dump)
        f_dump.close()
    
    def load(self,f_load):
        #f_load: file object
        #        laod parameters in the form of a dictionary from this file.
        #params of relevance: sigmoid and out_sigmoid's W and b
        # also, n_in and n_out for each.
        hidden_paramdicts = pickle.load(f_load)

        for (hidden_paramdict,layer) in zip(hidden_paramdicts,self.sigmoid_layers+self.out_sigmoid_layers):
            layer.W.set_value(hidden_paramdict['W'].astype('float32'))
            layer.b.set_value(hidden_paramdict['b'].astype('float32'))
            layer.n_in = hidden_paramdict['n_in']
            layer.n_out = hidden_paramdict['n_out']

        f_load.close()

    def __init__(
        self,
        numpy_rng,
        f_load_MLP=None,
        f_load_SDA=None,
        theano_rng=None,
        n_ins=784,
        hidden_layers_sizes=[500, 500],
        n_outs=10,
        corruption_levels=[0.1, 0.1],
        name_appendage='',
        xtropy_fraction = 0
    ):
        """ This class is made to support a variable number of layers.

        :type numpy_rng: numpy.random.RandomState
        :param numpy_rng: numpy random number generator used to draw initial
                    weights

        :type theano_rng: theano.tensor.shared_randomstreams.RandomStreams
        :param theano_rng: Theano random generator; if None is given one is
                           generated based on a seed drawn from `rng`

        :type n_ins: int
        :param n_ins: dimension of the input to the sdA

        :type n_layers_sizes: list of ints
        :param n_layers_sizes: intermediate layers size, must contain
                               at least one value

        :type n_outs: int
        :param n_outs: dimension of the output of the network

        :type corruption_levels: list of float
        :param corruption_levels: amount of corruption to use for each
                                  layer
        """

        self.sigmoid_layers = []
        self.out_sigmoid_layers = []
        self.dA_layers = []
        self.params = []
        self.n_layers = len(hidden_layers_sizes)

        assert self.n_layers > 0

        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30)) 
        # allocate symbolic variables for the data
        self.x = T.matrix('x')  # the data is presented as rasterized images
        self.y = T.ivector('y')  # the labels are presented as 1D vector of
                                 # [int] labels


        for i in xrange(self.n_layers):
            # the size of the input is either the number of hidden units of
            # the layer below or the input size if we are on the first layer
            if i == 0:
                input_size = n_ins
            else:
                input_size = hidden_layers_sizes[i - 1]

            # the input to this layer is either the activation of the hidden
            # layer below or the input of the SdA if you are on the first
            # layer
            if i == 0:
                layer_input = self.x
            else:
                layer_input = self.sigmoid_layers[-1].output

            sigmoid_layer = HiddenLayer_ReLU(rng=numpy_rng,
                                        input=layer_input,
                                        n_in=input_size,
                                        n_out=hidden_layers_sizes[i],
                                        name_appendage = name_appendage+'_sigmoid_'+str(i))
            
            # add the layer to our list of layers
            self.sigmoid_layers.append(sigmoid_layer)

            # its arguably a philosophical question...
            # but we are going to only declare that the parameters of the
            # sigmoid_layers are parameters of the StackedDAA
            # the visible biases in the dA are parameters of those
            # dA, but not the SdA
            self.params.extend(sigmoid_layer.params)
        
        for i in xrange(self.n_layers):
            all_layers = self.sigmoid_layers+self.out_sigmoid_layers

            input_size = all_layers[-1].n_out

            output_size = self.sigmoid_layers[-i-1].n_in

            # the input to the inverse sigmoid layer is always the activation of the
            # sigmoid layer behind it (forward sigmoid if its' the first inverse layer)
            layer_input = all_layers[-1].output
                
            out_sigmoid_layer = HiddenLayer_ReLU(rng=numpy_rng,
                                            input=layer_input,
                                            n_in=input_size,
                                            n_out=output_size,
                                            name_appendage = name_appendage+'_outsigmoid_'+str(i))
            
            self.out_sigmoid_layers.append(out_sigmoid_layer)
            self.params.extend(out_sigmoid_layer.params)
        
        for i in xrange(self.n_layers):
            sigmoid_layer = self.sigmoid_layers[i]                           
            # Construct a denoising autoencoder that shared weights with each layer
            dA_layer = dA(numpy_rng=numpy_rng,
                          theano_rng=theano_rng,
                          input=sigmoid_layer.input,
                          n_visible=sigmoid_layer.n_in,
                          n_hidden=sigmoid_layer.n_out,
                          W = sigmoid_layer.W,
                          bhid=sigmoid_layer.b,                 
                          name_appendage=name_appendage+'_dA_'+str(i)
                          )
            self.dA_layers.append(dA_layer)

        if f_load_MLP != None:
            self.predictLayer = MLP(
                rng = numpy_rng,
                input=self.out_sigmoid_layers[-1].output,
                f_load = f_load_MLP,
                name_appendage = name_appendage+'_MLPLayer'
                )
        elif f_load_SDA != None:
            self.predictLayer = SdA(
                numpy_rng = numpy_rng,
                n_ins=28 * 28,
                hidden_layers_sizes=[1000, 1000, 1000],
                n_outs=10,
                input = self.out_sigmoid_layers[-1].output
                )
            self.predictLayer.load(f_load_SDA)
            
        self.xtropy_cost = -T.mean(self.x*T.log(self.out_sigmoid_layers[-1].output) + (1-self.x)*T.log(1-self.out_sigmoid_layers[-1].output))
        self.mse_cost = T.mean((self.x-self.out_sigmoid_layers[-1].output)**2)
        self.logloss_cost = self.predictLayer.logLayer.negative_log_likelihood(self.y)
        self.finetune_cost = xtropy_fraction*self.mse_cost + (1-xtropy_fraction)*self.logloss_cost

        self.errors = self.predictLayer.logLayer.errors(self.y)


    def pretraining_functions(self, train_set_x, batch_size):
        ''' Generates a list of functions, each of them implementing one
        step in trainnig the dA corresponding to the layer with same index.
        The function will require as input the minibatch index, and to train
        a dA you just need to iterate, calling the corresponding function on
        all minibatch indexes.

        :type train_set_x: theano.tensor.TensorType
        :param train_set_x: Shared variable that contains all datapoints used
                            for training the dA

        :type batch_size: int
        :param batch_size: size of a [mini]batch

        :type learning_rate: float
        :param learning_rate: learning rate used during training for any of
                              the dA layers
        '''

        # index to a [mini]batch
        index = T.lscalar('index')  # index to a minibatch
        corruption_level = T.scalar('corruption')  # % of corruption to use
        learning_rate = T.scalar('lr')  # learning rate to use
        # begining of a batch, given `index`
        batch_begin = index * batch_size
        # ending of a batch given `index`
        batch_end = batch_begin + batch_size

        pretrain_fns = []
        for dA in self.dA_layers:
            # get the cost and the updates list
            cost, updates = dA.get_cost_updates_ReLU(corruption_level,
                                                learning_rate)
            # compile the theano function
            fn = theano.function(
                inputs=[
                    index,
                    theano.Param(corruption_level, default=0.2),
                    theano.Param(learning_rate, default=0.1)
                ],
                outputs=cost,
                updates=updates,
                givens={
                    self.x: train_set_x[batch_begin: batch_end]
                }
            )
            # append `fn` to the list of functions
            pretrain_fns.append(fn)

        return pretrain_fns

    def build_finetune_functions(self, datasets, batch_size, learning_rate):
        '''Generates a function `train` that implements one step of
        finetuning, a function `validate` that computes the error on
        a batch from the validation set, and a function `test` that
        computes the error on a batch from the testing set

        :type datasets: list of pairs of theano.tensor.TensorType
        :param datasets: It is a list that contain all the datasets;
                         the has to contain three pairs, `train`,
                         `valid`, `test` in this order, where each pair
                         is formed of two Theano variables, one for the
                         datapoints, the other for the labels

        :type batch_size: int
        :param batch_size: size of a minibatch

        :type learning_rate: float
        :param learning_rate: learning rate used during finetune stage
        '''

        (train_set_x, train_set_y) = datasets[0]
        (valid_set_x, valid_set_y) = datasets[1]
        (test_set_x, test_set_y) = datasets[2]

        # compute number of minibatches for training, validation and testing
        n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
        n_valid_batches /= batch_size
        n_test_batches = test_set_x.get_value(borrow=True).shape[0]
        n_test_batches /= batch_size

        index = T.lscalar('index')  # index to a [mini]batch

        # compute the gradients with respect to the model parameters
        gparams = T.grad(self.finetune_cost, self.params)

        # compute list of fine-tuning updates
        updates = [
            (param, param - gparam * learning_rate)
            for param, gparam in zip(self.params, gparams)
        ]

        pdb.set_trace()

        train_fn = theano.function(
            inputs=[index],
            outputs=self.finetune_cost,
            updates=updates,
            givens={
                self.x: train_set_x[
                    index * batch_size: (index + 1) * batch_size
                ],
                self.y: train_set_y[
                    index * batch_size: (index + 1) * batch_size
                ]
            },
            name='train'
        )

        test_score_i = theano.function(
            [index],
            self.errors,
            givens={
                self.x: test_set_x[
                    index * batch_size: (index + 1) * batch_size
                ],
                self.y: test_set_y[
                    index * batch_size: (index + 1) * batch_size
                ]
            },
            name='test'
        )

        valid_score_i = theano.function(
            [index],
            self.errors,
            givens={
                self.x: valid_set_x[
                    index * batch_size: (index + 1) * batch_size
                ],
                self.y: valid_set_y[
                    index * batch_size: (index + 1) * batch_size
                ]
            },
            name='valid'
        )
        
        valid_xtropy_logloss_i =  theano.function(
            inputs=[index],
            outputs=[self.xtropy_cost, self.logloss_cost],
            givens={
                self.x: valid_set_x[
                    index * batch_size: (index + 1) * batch_size
                ],
                self.y: valid_set_y[
                    index * batch_size: (index + 1) * batch_size
                ]
            },
            name='valid_xtropy_logloss'
        )
        # Create a function that scans the entire validation set
        def valid_score():
            return [valid_score_i(i) for i in xrange(n_valid_batches)]

        def valid_xtropy_logloss():
            return [valid_xtropy_logloss_i(i) for i in xrange(n_valid_batches)]

        # Create a function that scans the entire test set
        def test_score():
            return [test_score_i(i) for i in xrange(n_test_batches)]

        return train_fn, valid_score, test_score, valid_xtropy_logloss


def test_ssDA(finetune_lr=0.1, pretraining_epochs=15,
             pretrain_lr=0.001, training_epochs=1000,
             dataset='mnist.pkl.gz', batch_size=1):
    """
    Demonstrates how to train and test a stochastic denoising autoencoder.

    This is demonstrated on MNIST.

    :type learning_rate: float
    :param learning_rate: learning rate used in the finetune stage
    (factor for the stochastic gradient)

    :type pretraining_epochs: int
    :param pretraining_epochs: number of epoch to do pretraining

    :type pretrain_lr: float
    :param pretrain_lr: learning rate to be used during pre-training

    :type n_iter: int
    :param n_iter: maximal number of iterations ot run the optimizer

    :type dataset: string
    :param dataset: path the the pickled dataset

    """
    xtropy_fraction = 0
    dir_pretrained = '../data/train_snapshots/stacked_sda/'
    path_finetuned_pre = '../data/train_snapshots/stacked_sda/stackedSDA_pretrainedxtropy.p'#'/Users/vmisra/data/deepCompress_data/stackedSDA_xtropy1params.p'#../data/train_snapshots/stacked_sda/stackedSDA_pretrainedxtropy.p'
    path_finetuned_post = '../data/train_snapshots/stacked_sda/stackedSDA_prextropy1_postxtropy0_B.p'#'/Users/vmisra/data/deepCompress_data/stackedSDA_prextropy1_postxtropy0_B.p'#../data/train_snapshots/stacked_sda/stackedSDA_prextropy1_postxtropy0.p'

    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]
    
    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_train_batches /= batch_size

    # numpy random generator
    # start-snippet-3
    numpy_rng = numpy.random.RandomState(89677)
    print '... building the model'
    # construct the stacked denoising autoencoder class
    ssda = ssDA(
        numpy_rng=numpy_rng,
        n_ins=28 * 28,
        hidden_layers_sizes=[1000, 1000, 1000, 15],
        f_load_SDA = open('../data/Stacked_DA_params.p','r'),
        xtropy_fraction=xtropy_fraction
    )
    #########################
    # PRETRAINING THE MODEL #
    #########################
    print '... getting the pretraining functions'
    pretraining_fns = ssda.pretraining_functions(train_set_x=train_set_x,
                                                batch_size=batch_size)

    print '... pre-training the model'
    start_time = time.clock()
    ## Pre-train layer-wise
    corruption_levels = [.1, .2, .3, .3]#[0] #[.1, .2, .3]

    for i in xrange(ssda.n_layers):
        layerpath = dir_pretrained+ 'layer'+str(i)+'_snapshot_stacked_sda.p'

        if os.path.isfile(layerpath):
           ssda.load(open(layerpath,'r'))
           continue

        # go through pretraining epochs
        for epoch in xrange(pretraining_epochs):

            # go through the training set
            c = []
            for batch_index in xrange(n_train_batches):
                c.append(pretraining_fns[i](index=batch_index,
                         corruption=corruption_levels[i],
                         lr=pretrain_lr))
            print 'Pre-training layer %i, epoch %d, cost ' % (i, epoch),
            print numpy.mean(c)
            
        #COPY OVER PRETRAINED PARAMS FROM DA'S TO HIDDEN SIGMOIDS
        ssda.sigmoid_layers[i].W.set_value(ssda.dA_layers[i].W.eval())
        ssda.sigmoid_layers[i].b.set_value(ssda.dA_layers[i].b.eval())
        ssda.out_sigmoid_layers[-i-1].W.set_value(ssda.dA_layers[i].W.T.eval())
        ssda.out_sigmoid_layers[-i-1].b.set_value(ssda.dA_layers[i].b_prime.get_value())
        
        #dump snapshot
        ssda.dump(open(layerpath,'w'))

    end_time = time.clock()

    print >> sys.stderr, ('The pretraining code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))

    ########################
    # FINETUNING THE MODEL
    ########################
    #pre-load partially finetuned version, if it exists
    if os.path.isfile(path_finetuned_pre):
        ssda.load(open(path_finetuned_pre,'r'))
    
    # get the training, validation and testing function for the model
    print '... getting the finetuning functions'
    train_fn, validate_model, test_model, valid_xtropy_logloss = ssda.build_finetune_functions(
        datasets=datasets,
        batch_size=batch_size,
        learning_rate=finetune_lr
    )
    
    print '... finetuning the model'

    # early-stopping parameters
    patience = 10 * n_train_batches  # look as this many examples regardless
    patience_increase = 2.  # wait this much longer when a new best is
                            # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = numpy.inf
    test_score = 0.
    start_time = time.clock()

    done_looping = False
    epoch = 0

    while (epoch < training_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):
            minibatch_avg_cost = train_fn(minibatch_index)
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                validation_losses = validate_model()
                this_validation_loss = numpy.mean(validation_losses)
#                xtropy_logloss_loss = valid_xtropy_logloss()
#                xtropy_loss = [x[0] for x in xtropy_logloss_loss]
                print('epoch %i, minibatch %i/%i, validation error %f  %%' %
                      (epoch, minibatch_index + 1, n_train_batches,
                       this_validation_loss * 100., ))
                ssda.dump(open(path_finetuned_post,'w'))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    #improve patience if loss improvement is good enough
                    if (
                        this_validation_loss < best_validation_loss *
                        improvement_threshold
                    ):
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = test_model()
                    test_score = numpy.mean(test_losses)
                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))
        

            if patience <= iter:
                done_looping = True
                break

    end_time = time.clock()
    print(
        (
            'Optimization complete with best validation score of %f %%, '
            'on iteration %i, '
            'with test performance %f %%'
        )
        % (best_validation_loss * 100., best_iter + 1, test_score * 100.)
    )
    print >> sys.stderr, ('The training code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))
    ssda.dump(open(path_finetuned_post,'w'))

    return ssda

def test_ssDA_nopretraining(finetune_lr=0.1, pretraining_epochs=15,
             pretrain_lr=0.001, training_epochs=1000,
             dataset='mnist.pkl.gz', batch_size=1,
             data_dir = '../data/'):
    xtropy_fraction = 1
    path_finetuned_pre = os.path.join(data_dir,'train_snapshots/stacked_sda/stackedSDA_nopretrained_ReLU.p')
    path_finetuned_post = os.path.join(data_dir,'train_snapshots/stacked_sda/stackedSDA_nopretrained_ReLU_post.p')
    path_stacked_da = os.path.join(data_dir,'Stacked_DA_params.p')

    datasets = load_data(os.path.join(data_dir,dataset))

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]
    
    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_train_batches /= batch_size

    # numpy random generator
    # start-snippet-3
    numpy_rng = numpy.random.RandomState(89677)
    print '... building the model'
    # construct the stacked denoising autoencoder class
    ssda = ssDA(
        numpy_rng=numpy_rng,
        n_ins=28 * 28,
        hidden_layers_sizes=[1000, 1000, 1000, 15],
        f_load_SDA = open(path_stacked_da,'r'),
        xtropy_fraction=xtropy_fraction
    )

    #finetune training
    #pre-load partially finetuned version, if it exists
    if os.path.isfile(path_finetuned_pre):
        ssda.load(open(path_finetuned_pre,'r'))
    
    # get the training, validation and testing function for the model
    print '... getting the finetuning functions'
    train_fn, validate_model, test_model, valid_xtropy_logloss = ssda.build_finetune_functions(
        datasets=datasets,
        batch_size=batch_size,
        learning_rate=finetune_lr
    )
    
    print '... finetuning the model'

    # early-stopping parameters
    patience = 10 * n_train_batches  # look as this many examples regardless
    patience_increase = 2.  # wait this much longer when a new best is
                            # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = numpy.inf
    test_score = 0.
    start_time = time.clock()

    done_looping = False
    epoch = 0

    while (epoch < training_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):
            minibatch_avg_cost = train_fn(minibatch_index)
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                validation_losses = validate_model()
                this_validation_loss = numpy.mean(validation_losses)
#                xtropy_logloss_loss = valid_xtropy_logloss()
#                xtropy_loss = [x[0] for x in xtropy_logloss_loss]
                print('epoch %i, minibatch %i/%i, validation error %f  %%' %
                      (epoch, minibatch_index + 1, n_train_batches,
                       this_validation_loss * 100., ))
                ssda.dump(open(path_finetuned_post,'w'))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    #improve patience if loss improvement is good enough
                    if (
                        this_validation_loss < best_validation_loss *
                        improvement_threshold
                    ):
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = test_model()
                    test_score = numpy.mean(test_losses)
                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))
        

            if patience <= iter:
                done_looping = True
                break

    end_time = time.clock()
    print(
        (
            'Optimization complete with best validation score of %f %%, '
            'on iteration %i, '
            'with test performance %f %%'
        )
        % (best_validation_loss * 100., best_iter + 1, test_score * 100.)
    )
    print >> sys.stderr, ('The training code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))
    ssda.dump(open(path_finetuned_post,'w'))

    return ssda

if __name__ == '__main__':
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    else:
        data_dir = '/Users/vmisra/data/deepCompress_data/'
    ssda = test_ssDA_nopretraining(finetune_lr=0.01, batch_size=10, data_dir = data_dir)
