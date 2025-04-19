import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.metrics import classification_report, confusion_matrix
import random
import os
import h5py

#              DLModel
# =============================================================

class DLModel:
    def __init__(self, name="Model"):
        self.name = name
        self.layers = [None]
        self._is_compiled = False
        self.threshold = 0.7
        self.scale_random = 0.01
        self.forced_num_classess = 0

    # add a layer to the model
    def add(self, layer):
        self.layers.append(layer)

    # loss functions
    # --------------
    def squared_means(self, AL, Y):
        m = Y.shape[1]
        return np.power(AL-Y, 2) / m
    def squared_means_derivative(self, AL, Y):
        m = Y.shape[1]
        return 2*(AL-Y) / m
    def cross_entropy(self, AL, Y):
        return np.where(Y == 0, -np.log(1-AL), -np.log(AL))/Y.shape[1]
    def cross_entropy_derivative(self, AL, Y):
        return np.where(Y == 0, 1/(1-AL), -1/AL)/Y.shape[1]

    # compile the model. must be called prior to training
    def compile(self, loss, threshold = 0.7):
        self._is_compiled = True
        self.loss = loss
        self.threshold = threshold

        if (loss == "squared_means"):
            self.loss_forward = self.squared_means
            self.loss_backward = self.squared_means_derivative
        elif (loss == "cross_entropy"):
            self.loss_forward = self.cross_entropy
            self.loss_backward = self.cross_entropy_derivative
        elif (loss == "categorical_cross_entropy"):
            self.loss_forward = self._categorical_cross_entropy
            self.loss_backward = self._categorical_cross_entropy_backward
        else:
            print("*** Invalid loss function")
            raise NotImplementedError("Unimplemented loss function: " + loss)

    # compute the cost
    def compute_cost(self, AL, Y):
        return np.sum(self.loss_forward(AL, Y))

    # train the model
    # ---------------
    def train(self, X, Y, num_iterations):
        print_ind = max(num_iterations // 100, 1)       # print progress every 1% of the iterations
        L = len(self.layers)
        costs = []

        for i in range(num_iterations):
            # forward propagation
            Al = X
            for l in range(1,L):
                Al = self.layers[l].forward_propagation(Al,False)
            #backward propagation
            dAl = self.loss_backward(Al, Y)

            for l in reversed(range(1,L)):
                dAl = self.layers[l].backward_propagation(dAl)
                # update parameters
                self.layers[l].update_parameters()

            #record progress
            if i > 0 and i % print_ind == 0:
                J = self.compute_cost(Al, Y)
                costs.append(J)
                print("cost after ",str(i//print_ind),"%:",str(J))

        costs.append(self.compute_cost(Al, Y))  # record the last cost
        return costs

    # predicts the value of a new sample
    def predict(self, X):
       print (" ====##########=== In Predict ====##########################===")
       L = len(self.layers)
       Al = X
       for l in range(1, L):
           Al = self.layers[l].forward_propagation(Al, True)
       if self.layers[-1]._activation == "softmax":
           # For softmax activation, return 1 for the max value in each column and 0 otherwise
           prediction = (Al == Al.max(axis=0, keepdims=True)).astype(int)
           print (f" Return Val Predict = {prediction} ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
           return (Al == Al.max(axis=0, keepdims=True)).astype(int)
       else:
           return np.where(Al > self.threshold, 1, 0)


    def __str__(self):
        s = self.name + " description:\n\tnum_layers: " + str(len(self.layers)-1) +"\n"
        if self._is_compiled:
            s += "\tCompilation parameters:\n"
            s += "\t\tprediction threshold: " + str(self.threshold) +"\n"
            s += "\t\tloss function: " + self.loss + "\n\n"
        for i in range(1,len(self.layers)):
            s += "\tLayer " + str(i) + ":" + str(self.layers[i]) + "\n"
        return s

    #save the weights of the model. Acitate the save_weights of each layer
    def save_weights(self,path):
        for i in range(1,len(self.layers)):
            self.layers[i].save_weights(path,"Layer"+str(i))

    def _categorical_cross_entropy(self, AL, Y):
        m = Y.shape[1]
        return -np.sum(Y * np.log(AL)) / m

    def _categorical_cross_entropy_backward(self, AL, Y):
        dZl = AL - Y
        return dZl

    def confusion_matrix(self, X, Y):
        prediction = self.predict(X)
        prediction_index = np.argmax(prediction, axis=0)
        Y_index = np.argmax(Y, axis=0)
        right = np.sum(prediction_index == Y_index)
        print("accuracy: ",str(right/len(Y[0])))
        cf = confusion_matrix(prediction_index, Y_index)
        print(cf)
        return cf

# =============================================================
# =============================================================
#              DLLayer
# =============================================================
# =============================================================
# This class implements a one layer of nuorons (Perceptrons).
# Input / Internal parameters:
# name - A string for the ANN (model)
# num_units - number of nuorons in the layer
# input_shape - number of inputs that get into the layer
# activation - name of the activation function (same for all the layer). implemented:
#    - sigmoid
#    - trim_sigmoid
#    - tanh
#    - trim_tanh
#    - relu ( default )
#    - leaky_relu
#    - softmax
#    - NoActivation
# W_initialization - name of the initialization funciton (same for all the layer), implemented : zeros, random.
# learning_rate - sometimes called alpha.
# optimization - the algorithm to use for the gradient descent parameters update (e.g. adaptive)
#
# Algorithm:
#    * Forward and Backward propagation

class DLLayer:
    def __init__(self, name, num_units, input_shape : tuple, activation="relu", W_initialization = "random", learning_rate = 1.2, optimization=None, random_scale = 0.01):
        self.name = name
        self.alpha = learning_rate
        self._num_units = num_units
        self._input_shape = input_shape
        self._activation = activation
        self.prediction_function = activation
        self._optimization = optimization

        self._random_scale = random_scale
        self.activation_trim = 1e-10
        self._activation_forward = activation;

        if (activation == "leaky_relu"):
            self.leaky_relu_d = 0.01 # default value

        if (optimization == "adaptive"):
            self._adaptive_alpha_b = np.full((self._num_units, 1), self.alpha)
            self._adaptive_alpha_W = np.full((self._num_units, self._input_shape[0]), self.alpha)
            self.adaptive_cont = 1.1
            self.adaptive_switch = 0.5

        self.init_weights(W_initialization)

        # activation methods
        if activation == "sigmoid":
            self.activation_forward = self._sigmoid
            self.activation_backward = self._sigmoid_backward
        elif activation == "trim_sigmoid":
            self.activation_forward = self._trim_sigmoid
            self.activation_backward = self._trim_sigmoid_backward
        elif activation == "tanh":
            self.activation_forward = self._tanh
            self.activation_backward = self._tanh_backward
        elif activation == "trim_tanh":
            self.activation_forward = self._trim_tanh
            self.activation_backward = self._trim_tanh_backward
        elif activation == "relu":
            self.activation_forward = self._relu
            self.activation_backward = self._relu_backward
        elif activation == "leaky_relu":
            self.activation_forward = self._leaky_relu
            self.activation_backward = self._leaky_relu_backward
        elif activation == "softmax":
            self.activation_forward = self._softmax
            self.activation_backward = self._softmax_backward
        elif activation == "trim softmax":
            self.activation_forward = self._trim_softmax
            self.activation_backward = self._softmax_backward
        else:
            self.activation_forward = None
            self.activation_backward = None
            print("*** Invalid activation type")
            raise NotImplementedError("Unrecognized activation function:", activation)
        self._prediction=self.activation_forward

    def _get_W_shape(self):
        return (self._num_units, *(self._input_shape))

    def init_weights(self, W_initialization):
        self.b = np.zeros((self._num_units,1), dtype=float) # b is init to zeros, always
        if (W_initialization == "random"):
            self.W = np.random.randn(self._num_units, *(self._input_shape)) * self._random_scale
        elif (W_initialization == "zeros"):
            self.W = np.zeros((self._num_units, *self._input_shape), dtype=float)
        elif W_initialization == "He":
            self.W = np.random.randn(*self._get_W_shape()) * np.sqrt(2.0/sum(self._input_shape))
        elif W_initialization == "Xaviar":
            self.W = np.random.randn(*self._get_W_shape()) * np.sqrt(1.0/sum(self._input_shape))
        else:
            try:
                with h5py.File(W_initialization, 'r') as hf:
                    self.W = hf['W'][:]
                    self.b = hf['b'][:]
                    print(f"((((((((((((((((((( Found H5 File {W_initialization} )))))))")
            except (FileNotFoundError):
                current_directory = os.getcwd()
                print("###################################################################Current Working Directory:", current_directory)
                print(f"((((((((((((((((((( NOT NOT Found H5 File {W_initialization} )))))))")
                raise NotImplementedError("Unrecognized initialization:", W_initialization)

    def __str__(self):
        s = self.name + " Layer:\n"
        s += "\tnum_units: " + str(self._num_units) + "\n"
        s += "\tactivation: " + self._activation + "\n"
        if self._activation == "leaky_relu":
            s += "\t\tleaky relu parameters:\n"
            s += "\t\t\tleaky_relu_d: " + str(self.leaky_relu_d)+"\n"
        s += "\tinput_shape: " + str(self._input_shape) + "\n"
        s += "\tlearning_rate (alpha): " + str(self.alpha) + "\n"
        #optimization
        if self._optimization == "adaptive":
            s += "\t\tadaptive parameters:\n"
            s += "\t\t\tcont: " + str(self.adaptive_cont)+"\n"
            s += "\t\t\tswitch: " + str(self.adaptive_switch)+"\n"

        # parameters
        s += "\tparameters:\n\t\tb.T: " + str(self.b.T) + "\n"
        s += "\t\tshape weights: " + str(self.W.shape)+"\n"
        plt.hist(self.W.reshape(-1))
        plt.title("W histogram")
        plt.show()
        return s

    # activation functions - forwars and backward
    # -------------------------------------------
    def _sigmoid(self, Z):
        return 1/(1+np.exp(-Z))
    def _sigmoid_backward(self,dA):
        A = self._sigmoid(self.Z)
        dZ = dA * A * (1-A)
        return dZ

    def _relu(self, Z):
        return np.maximum(0,Z)
    def _relu_backward(self,dA):
        dZ = np.where(self.Z <= 0, 0, dA)
        return dZ

    def _leaky_relu(self, Z):
        return np.maximum(self.leaky_relu_d*Z,Z)
    def _leaky_relu_backward(self,dA):
        dZ = np.where(self.Z <= 0, self.leaky_relu_d * dA, dA)
        return dZ

    def _tanh(self, Z):
        return np.tanh(Z)
    def _tanh_backward(self,dA):
        dZ = dA * (1 - np.power(self._tanh(self.Z), 2))
        return dZ

    def _trim_sigmoid(self,Z):
        with np.errstate(over='raise', divide='raise'):
            try:
                A = 1/(1+np.exp(-Z))
            except FloatingPointError:
                Z = np.where(Z < -100, -100, Z)
                A = 1/(1+np.exp(-Z))
        TRIM = self.activation_trim
        if TRIM > 0:
            A = np.where(A < TRIM,TRIM,A)
            A = np.where(A > 1-TRIM,1-TRIM, A)
        return A
    def _trim_sigmoid_backward(self,dA):
        A = self._trim_sigmoid(self.Z)
        dZ = dA * A * (1-A)
        return dZ

    def _trim_tanh(self,Z):
        A = np.tanh(Z)
        TRIM = self.activation_trim
        if TRIM > 0:
            A = np.where(A < -1+TRIM,TRIM,A)
            A = np.where(A > 1-TRIM,1-TRIM, A)
        return A
    def _trim_tanh_backward(self,dA):
        A = self._trim_tanh(self._Z)
        dZ = dA * (1-A**2)
        return dZ



    # forward propagation
    # -------------------
    def forward_propagation(self, A_prev, is_predict):
        self._A_prev = A_prev
        self.Z = np.dot(self.W, self._A_prev) + self.b
        if (is_predict):
            self.A = self._prediction(self.Z)
        else:
            self.A = self.activation_function(self.Z)
        return self.A


    # backword propagation
    # --------------------
    def backward_propagation(self, dA):
        m = self._A_prev.shape[1]
        dZ = self.activation_backward(dA)
        self.dW = np.dot(dZ, self._A_prev.T) / m
        self.db = np.sum(dZ, axis=1, keepdims=True) / m
        dA_prev = np.dot(self.W.T, dZ)
        return dA_prev

    # update parameters
    # -----------------
    def update_parameters(self):
        if self._optimization == "adaptive":    # Update parameters with adaptive learning rate. keep the sign positive. Update is multiply by the derived value
            self._adaptive_alpha_b = np.where(self.db * self._adaptive_alpha_b >= 0, self._adaptive_alpha_b * self.adaptive_cont, self._adaptive_alpha_b * self.adaptive_switch)
            self._adaptive_alpha_W = np.where(self.dW * self._adaptive_alpha_W >= 0, self._adaptive_alpha_W * self.adaptive_cont, self._adaptive_alpha_W * self.adaptive_switch)
            self.W -= self._adaptive_alpha_W * self.dW
            self.b -= self._adaptive_alpha_b * self.db
        else:
            self.W -= self.alpha * self.dW
            self.b -= self.alpha * self.db


    # save the Ws and b of the layer
    def save_weights(self,path,file_name):
        if not os.path.exists(path):
            os.makedirs(path)

        with h5py.File(path+"/"+file_name+'.h5', 'w') as hf:
            hf.create_dataset("W",  data=self.W)
            hf.create_dataset("b",  data=self.b)

# =============================================================
    def _softmax(self, Z):
       expZ = np.exp(Z - np.max(Z))
       return expZ / expZ.sum(axis=0, keepdims=True)

    def _softmax_backward(self, dA):
        return dA

# =============================================================
    def _trim_softmax(self, Z):
        with np.errstate(over='raise', divide='raise'):
            try:
                eZ = np.exp(Z)
            except FloatingPointError:
                Z = np.where(Z > 100, 100,Z)
                eZ = np.exp(Z)
        A = eZ/np.sum(eZ, axis=0)
        return A

    def activation_function(self, Z):
            return self.activation_forward(Z)



