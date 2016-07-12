from keras import backend as K
from keras.engine.topology import Layer
from keras.layers import Input
from keras.models import Model

from gensim.models import Word2Vec

import numpy as np
import theano.tensor as T
from theano import function

import sys
import file_util


def tensor_illuminator(x, input):
    #das ist naturlich quatsch. hat auch wieder keine echten werte, sondern ist nur ein Tensor!
    return x.shape.eval(input)



class MagicOperation(Layer):
    def __init__(self, output_dim, composition_mode = 'tensor_mult', **kwargs):
        self.output_dim = output_dim
        self.composition_mode = composition_mode
        # self.composition_mode = composition_mode
        super(MagicOperation, self).__init__(**kwargs)

    def build(self, input_shape):
        # print input_shape
        input_dim = input_shape[2]
        initial_weight_value = 0
        if self.composition_mode == 'tensor_mult_random':
            initial_weight_value = np.random.random((input_dim, input_dim, input_dim))
            # print("Shape von initial weight values bei tensor_mult_random: {}".format(np.shape(initial_weight_value)))
        elif self.composition_mode == 'tensor_mult_identity':
            arr = []
            for i in range(0,input_dim):
                arr.append(np.identity(input_dim))
            initial_weight_value = np.array(arr)
            # print("Shape von initial weight values bei tensor_mult_identity: {}".format(np.shape(initial_weight_value)))
        elif self.composition_mode == 'weighted_adj_add_identity':
            initial_weight_value = np.identity(input_dim)
        elif self.composition_mode == 'weighted_noun_add_identity':
            initial_weight_value = np.identity(input_dim)
        elif self.composition_mode == 'weighted_adj_and_noun_add_identity':
            initial_weight_value = np.asarray([np.identity(input_dim).tolist(),
                                               np.identity(input_dim).tolist()])
        elif self.composition_mode == 'weighted_adj_add_random':
            initial_weight_value = np.random.random((input_dim,input_dim))
        elif self.composition_mode == 'weighted_noun_add_random':
            initial_weight_value = np.random.random((input_dim,input_dim))
        elif self.composition_mode == 'weighted_adj_and_noun_add_random':
            initial_weight_value = np.asarray([np.random.random((input_dim,input_dim)).tolist(),
                                               np.random.random((input_dim,input_dim)).tolist()])
        elif self.composition_mode == 'weighted_adj_add_ones':
            initial_weight_value = np.ones((input_dim,input_dim))
        elif self.composition_mode == 'weighted_noun_add_ones':
            initial_weight_value = np.ones((input_dim,input_dim))
        elif self.composition_mode == 'weighted_adj_and_noun_add_ones':
            initial_weight_value = np.asarray([np.ones((input_dim,input_dim)).tolist(),
                                               np.ones((input_dim,input_dim)).tolist()])
        elif self.composition_mode == 'weighted_adj_and_noun_add_identity_with_rands':
            W_one_dim = ((np.ones((input_dim,input_dim)) - np.identity(input_dim)) * np.random.random((input_dim,input_dim)) * 0.1 + np.identity((input_dim))).tolist()
            # W_noun = ((np.ones((input_dim,input_dim)) - np.identity(input_dim)) * np.random.random((input_dim,input_dim)) * 0.1 + np.identity((input_dim))).tolist()
            initial_weight_value = np.asarray([W_one_dim,W_one_dim])
        elif self.composition_mode == 'weighted_adj_noun_add_sum1_identity':
            initial_weight_value = np.identity(input_dim)
        elif self.composition_mode == 'weighted_adj_noun_add_sum1_random':
            initial_weight_value = np.random.random((input_dim,input_dim))
        self.W = K.variable(initial_weight_value)
        self.trainable_weights = [self.W]

    def call(self, x, mask=None):

        adj = x[:,0,:] #sind schon tensoren, wegen der Input-schicht davor!
        noun = x[:,1,:]

        attribute = []

        if 'tensor_mult' in self.composition_mode:
            # print("Call mit tensor_mult!")
            adj_matrix = T.tensordot(adj,self.W,[[1],[2]])
            attribute = T.tensordot(noun,adj_matrix, [[1],[1]])
        elif 'weighted_adj_add' in self.composition_mode:
            weighted_adj = T.dot(adj, self.W)
            attribute = weighted_adj + noun
        elif 'weighted_noun_add' in self.composition_mode:
            weighted_noun = T.dot(noun, self.W)
            attribute = adj + weighted_noun
        elif 'weighted_adj_and_noun_add' in self.composition_mode:
            weighted_adj = T.dot(adj, self.W[0])
            weighted_noun = T.dot(noun, self.W[1])
            attribute = weighted_adj + weighted_noun
        elif 'weighted_adj_noun_add_sum1' in self.composition_mode:
            weighted_adj= T.dot(adj, self.W)
            weighted_noun = T.dot(adj, 1 - self.W)
            attribute = weighted_adj + weighted_noun
        return attribute

    def get_output_shape_for(self, input_shape):
        # print input_shape
        return (input_shape[0], self.output_dim)

############AB HIER Trainingsprozess##############

   #konstruiere datenmatrix für training

def construct_data_and_labels(aan_list, vector_space, attr_train_set, verbosity = 2):
    #falls attr_train_set nicht leer, werden nur vektoren aus attribute subset genutzt
    if verbosity >= 2:
        print("Trainset in construct data: {}".format(attr_train_set))
        print("aan_list in construct data hat Länge {}".format(len(aan_list)))
    tmp_data = []
    tmp_labels = []
    not_in_embedding_space = []

    for aan in aan_list:
        adj = aan[1]
        noun = aan[2]
        attr = aan[0]

        if attr.upper() in attr_train_set:
            #nur falls das attribut in der konkreten Trainingsmenge enthalten ist
            if verbosity >= 2:
                print("{} ist in Trainings-set".format(attr.upper()))
            adjective_noun = []
            attribute = []

            x = False
            y = False

            try:
                adjective_noun = [vector_space[adj],vector_space[noun]]
                # adjective_noun = vector_space[adj] #nur zu testzwecken
                x = True
            except KeyError:
                if verbosity >=2:
                    print("Adjektiv oder Nomen beim Training nicht in WordEmbeddings enthalten: %s,%s" % (adj,noun))
                if adj not in not_in_embedding_space and noun not in not_in_embedding_space:
                    not_in_embedding_space += [adj,noun]



            try:
                attribute = vector_space[attr.lower()]
                y = True
            except KeyError:
                if verbosity>=2:
                    print("Attribut beim Training nicht in WordEmbeddings enthalten: %s" % (attr))
                if attr not in not_in_embedding_space:
                    not_in_embedding_space += [attr]



            if x and y:
                tmp_data.append(adjective_noun)
                tmp_labels.append(attribute)

    tmp_data = np.asarray(tmp_data)
    tmp_labels = np.asarray(tmp_labels)

    if verbosity >= 1 and np.shape(tmp_data) == (0,) or np.shape(tmp_labels) == (0,):
        print("Data oder Labels wurden nicht korrekt erstellt: shape == (0,)")


    return tmp_data,tmp_labels,not_in_embedding_space



def train_model(data, labels, composition_mode, verbosity=2):
    if verbosity >= 1:
        print("Trainiere NN mit mode %s..." % composition_mode)

    adj_noun_input = Input(shape=(2,300)) #2 vektoren, je 300 dimensionen

    output = MagicOperation(300, composition_mode = composition_mode)(adj_noun_input)

    model = Model(input = adj_noun_input, output = output)
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

    model.fit(data,labels, verbose=0)

    model.get_weights()

    return model


def min_mean_max_weight_matrix(weights):
    W = weights[0]

    W_adj = W[0].tolist()
    W_noun = W[1].tolist()

    result = [(np.min(W_adj),np.mean(W_adj),np.max(W_adj)), (np.min(W_noun), np.mean(W_noun), np.max(W_noun))]

    return result