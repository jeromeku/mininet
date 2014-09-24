import cPickle
import gzip
import tempfile
import os

import numpy as np
import time
import theano
import theano.tensor as T
from theano.compat.python2x import OrderedDict

def load_data(dataset):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the name of the dataset (here MNIST)
    '''

    # Check if dataset is in the data directory.
    data_path = os.path.join(os.path.split(__file__)[0], "data")
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    data_file = os.path.join(data_path, dataset)
    if os.path.isfile(data_file):
        dataset = data_file

    if (not os.path.isfile(data_file)) and dataset == 'mnist.pkl.gz':
        import urllib
        origin = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        print 'Downloading data from %s' % origin
        urllib.urlretrieve(origin, data_file)

    print '... loading data'

    # Load the dataset
    f = gzip.open(data_file, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()

    test_set_x, test_set_y = test_set
    valid_set_x, valid_set_y = valid_set
    train_set_x, train_set_y = train_set

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval


class MininetBase(object):
    def __getstate__(self):
        if not hasattr(self, '_pickle_skip_list'):
            self._pickle_skip_list = []
            for k, v in self.__dict__.items():
                try:
                    f = tempfile.TemporaryFile()
                    cPickle.dump(v, f)
                except:
                    self._pickle_skip_list.append(k)
        state = OrderedDict()
        for k, v in self.__dict__.items():
            if k not in self._pickle_skip_list:
                state[k] = v
        return state

    def __setstate__(self, state):
        self.__dict__ = state


class LogisticRegression(MininetBase):
    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, input, n_in, n_out):
        """ Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        """

        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        self.W = theano.shared(value=np.zeros((n_in, n_out),
                                              dtype=theano.config.floatX),
                               name='W', borrow=True)
        # initialize the baises b as a vector of n_out 0s
        self.b = theano.shared(value=np.zeros((n_out,),
                                              dtype=theano.config.floatX),
                               name='b', borrow=True)

        # compute vector of class-membership probabilities in symbolic form
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        # compute prediction as class whose probability is maximal in
        # symbolic form
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        # parameters of the model
        self.params = [self.W, self.b]

    def negative_log_likelihood(self, y):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """

        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError('y should have the same shape as self.y_pred')
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()


class HiddenLayer(MininetBase):

    def __init__(self, input, n_in, n_out, rng, W=None, b=None,
                 activation=T.tanh):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        self.input = input

        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for tanh activation function
        # the output of uniform if converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        # Note : optimal initialization of weights is dependent on the
        #        activation function used (among other things).
        #        For example, results presented in [Xavier10] suggest that you
        #        should use 4 times larger initial weights for sigmoid
        #        compared to tanh
        #        We have no info for other function, so we use the same as
        #        tanh.
        if W is None:
            W_values = np.asarray(rng.uniform(
                low=-np.sqrt(6. / (n_in + n_out)),
                high=np.sqrt(6. / (n_in + n_out)),
                size=(n_in, n_out)), dtype=theano.config.floatX)
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = np.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (lin_output if activation is None
                       else activation(lin_output))
        # parameters of the model
        self.params = [self.W, self.b]


class MLP(MininetBase):

    """Multi-Layer Perceptron Class

    A multilayer perceptron is a feedforward artificial neural network model
    that has one layer or more of hidden units and nonlinear activations.
    Intermediate layers usually have as activation function tanh or the
    sigmoid function (defined here by a ``HiddenLayer`` class)  while the
    top layer is a softamx layer (defined here by a ``LogisticRegression``
    class).
    """

    def __init__(self, hidden_layer_sizes=[500], batch_size=20, max_iter=1E5,
                 learning_rate=0.01, l1_reg=0., l2_reg=1E-4, random_seed=None,
                 model_save_path=None):
        """Initialize the parameters for the multilayer perceptron

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type hidden_layer_sizes: list of int
        :param hidden_layer_sizes: number of units per hidden layer,
        the dimension of the space in which the labels lie.

        """
        if random_seed is None or type(random_seed) in [float, int]:
            self.random_state = np.random.RandomState(random_seed)

        self.max_iter = max_iter
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        if model_save_path is None:
            self.model_save_path = 'model.save'
        else:
            self.model_save_path = model_save_path
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.hidden_layer_sizes = hidden_layer_sizes

    def _setup_functions(self, input_size, output_size):
        X_sym = T.matrix('x')
        y_sym = T.ivector('y')
        self.hidden_layers_ = []

        sizes = [input_size]
        sizes.extend(self.hidden_layer_sizes)
        input_variable = X_sym
        for n_in, n_out in zip(sizes[:-1], sizes[1:]):
            self.hidden_layers_.append(HiddenLayer(rng=self.random_state,
                                                   input=input_variable,
                                                   n_in=n_in, n_out=n_out,
                                                   activation=T.tanh))
            input_variable = self.hidden_layers_[-1].output

        self.output_layer_ = LogisticRegression(input=input_variable,
                                                n_in=sizes[-1],
                                                n_out=output_size)

        # L1 norm ; one regularization option is to enforce L1 norm to
        # be small
        self.l1 = abs(self.output_layer_.W).sum()
        for hl in self.hidden_layers_:
            self.l1 += abs(hl.W).sum()

        # square of L2 norm ; one regularization option is to enforce
        # square of L2 norm to be small
        self.l2_sqr = (self.output_layer_.W ** 2).sum()
        for hl in self.hidden_layers_:
            self.l2_sqr += (hl.W ** 2).sum()

        # negative log likelihood of the MLP is given by the negative
        # log likelihood of the output of the model, computed in the
        # logistic regression layer
        self.negative_log_likelihood = self.output_layer_.negative_log_likelihood
        # same holds for the function computing the number of errors
        self.errors = self.output_layer_.errors

        self.params = self.output_layer_.params
        for hl in self.hidden_layers_:
            self.params += hl.params
        self.cost = self.negative_log_likelihood(y_sym)
        self.cost += self.l1_reg * self.l1
        self.cost += self.l2_reg * self.l2_sqr

        self.gparams = []
        for param in self.params:
            gparam = T.grad(self.cost, param)
            self.gparams.append(gparam)

        self.updates = []
        for param, gparam in zip(self.params, self.gparams):
            self.updates.append((param,
                                 param - self.learning_rate * gparam))

        self.fit_function = theano.function(inputs=[X_sym, y_sym],
                                            outputs=self.cost,
                                            updates=self.updates)

        self.predict_function = theano.function(
            inputs=[X_sym], outputs=self.output_layer_.y_pred)

    def partial_fit(self, X, y):
        return self.fit_function(X, y.astype('int32'))

    def fit(self, X, y, valid_X=None, valid_y=None):
        self.input_size_ = X.shape[1]
        self.output_size_ = len(np.unique(y))
        if not hasattr(self, 'fit_function'):
            self._setup_functions(self.input_size_, self.output_size_)

        batch_indices = list(range(0, X.shape[0], self.batch_size))
        if X.shape[0] != batch_indices[-1]:
            batch_indices.append(X.shape[0])

        start_time = time.clock()
        itr = 0
        done_looping = False
        best_validation_score = np.inf
        patience = 10000
        patience_increase = 2
        improvement_threshold = 0.995
        while (itr < self.max_iter) and (not done_looping):
            print("Starting pass %d through the dataset" % itr)
            itr += 1
            for start, end in zip(batch_indices[:-1], batch_indices[1:]):
                self.partial_fit(X[start:end], y[start:end])

            if valid_X is None:
                # Save every iteration if no validation set
                f = file(self.model_save_path, 'wb')
                cPickle.dump(self, f, protocol=2)
                f.close()
            else:
                # Check early stopping criteria to see if we save or not
                assert valid_y is not None
                current_validation_score = (
                    self.predict(valid_X) != valid_y).mean()
                print("Validation score %f" % current_validation_score)
                # if we got the best validation score until now
                if current_validation_score < best_validation_score:
                    # improve patience if loss improvement is good enough
                    if current_validation_score < (best_validation_score *
                                                   improvement_threshold):
                        patience = max(patience, itr * len(batch_indices)
                                       * patience_increase)
                    print("Improved validation score %f" %
                          current_validation_score)
                    print("Previous best validation score %f" %
                          best_validation_score)
                    best_validation_score = current_validation_score
                    f = file(self.model_save_path, 'wb')
                    cPickle.dump(self, f, protocol=2)
                    f.close()
                if patience <= (itr * len(batch_indices)):
                        done_looping = True
        end_time = time.clock()
        print("Total training time ran for %.2fm" %
              ((end_time - start_time) / 60.))
        return self

    def predict(self, X):
        return self.predict_function(X)
