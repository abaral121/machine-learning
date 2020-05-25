# import all required libraries
import seaborn as sns; sns.set()
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons
from sklearn.datasets import load_digits
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from matplotlib.colors import ListedColormap
from sklearn import datasets
from sklearn import cluster
from sklearn import mixture
from sklearn import datasets
from sklearn.decomposition import PCA

import warnings
warnings.filterwarnings('ignore')

%matplotlib inline
%config InlineBackend.figure_format='retina'

class myNeuralNetwork(object):
    def __init__(self, n_in, n_layer1, n_layer2, n_out, learning_rate=0.1):

        self.n_in = n_in
        self.n_layer1 = n_layer1
        self.n_layer2 = n_layer2
        self.n_out = n_out
        self.learning_rate = learning_rate

        self.w1 = np.random.randn(self.n_layer1, self.n_in)
        self.w2 = np.random.randn(self.n_layer2, self.n_layer1)
        self.w3 = np.random.randn(self.n_out, self.n_layer2)
        pass


    def forward_propagation(self, x):

        self.x = x.reshape(-1, 1)

        self.z1 = self.w1 @ self.x
        self.a1 = self.sigmoid(self.z1)

        self.z2 = self.w2 @ self.a1
        self.a2 = self.sigmoid(self.z2)

        self.z3 = self.w3 @ self.a2
        self.a3 = self.sigmoid(self.z3)

        y_hat = self.a3

        return y_hat


    def compute_loss(self, X, y):

        errors = (y - self.predict_proba(X))**2
        mean_error = errors.mean()
        return mean_error


    def backpropagate(self, x, y):

        x = x.reshape(-1, 1)
        y = y.reshape(-1, 1)

        y_hat = self.forward_propagation(x)


        self.dE_dw3 = (y_hat - y) * (self.sigmoid(self.z3)) * (1 - self.sigmoid(self.z3)) @ (self.a2.T)
        self.dE_dw2 = ((self.w3.T @ ((y_hat - y) * (self.sigmoid(self.z3)) * (1 - self.sigmoid(self.z3)))) * ((self.sigmoid(self.z2)) * (1 - self.sigmoid(self.z2)))) @ self.a1.T
        self.dE_dw1 = ((self.w2.T @ (((self.w3.T @ ((y_hat - y) * (self.sigmoid(self.z3)) * (1 - self.sigmoid(self.z3)))) * ((self.sigmoid(self.z2)) * (1 - self.sigmoid(self.z2)))))) * ((self.sigmoid(self.z1)) * (1 - self.sigmoid(self.z1)))) @ self.x.T

        pass


    def stochastic_gradient_descent_step(self):

        self.w3 -= self.learning_rate * self.dE_dw3
        self.w2 -= self.learning_rate * self.dE_dw2
        self.w1 -= self.learning_rate * self.dE_dw1

        pass


    def fit(self, X, y, max_epochs=100, get_validation_loss=False, verbose=False):

        n_samples = X.shape[0]
        train_losses = []
        val_losses = []

        for epoch in range(max_epochs):

            shuffled_indices = np.random.choice(np.arange(X.shape[0]), size = X.shape[0], replace=False)
            X = X[shuffled_indices]
            y = y[shuffled_indices]


            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=4)


            for sample in range(X_train.shape[0]):

                self.forward_propagation(X_train[sample])
                self.backpropagate(X_train[sample], y_train[sample])
                self.stochastic_gradient_descent_step()
                pass


            train_losses.append(self.compute_loss(X_train, y_train))
            val_losses.append(self.compute_loss(X_val, y_val))


        if(get_validation_loss):
            return train_losses, val_losses
        else:
            return train_losses


    def predict_proba(self, X):

        z1 = self.w1 @ X.T
        a1 = self.sigmoid(z1)

        z2 = self.w2 @ a1
        a2 = self.sigmoid(z2)

        z3 = self.w3 @ a2
        y_hat = self.sigmoid(z3)

        return y_hat.flatten()


    def predict(self, X, decision_thresh=0.5):


        y_hat = self.predict_proba(X)


        y_hat[y_hat > decision_thresh]  = 1
        y_hat[y_hat <= decision_thresh] = 0

        return y_hat.flatten()


    def sigmoid(self, X):

        X_sigmoid = 1 / (1 + np.exp(-X))
        return X_sigmoid


    def sigmoid_derivative(self, X):

        return self.sigmoid(X) * (1 - self.sigmoid(X))


N_train = 500
X, y = make_moons(n_samples=N_train, noise=0.20)

N_test  = 100
X_test, y_test = make_moons(n_samples=N_test, noise=0.20)

num_epochs = 500
lr = 1
n_in = X.shape[1]
n_hidden_1 = 5
n_hidden_2 = 5
n_out = 1
nn = myNeuralNetwork(n_in=n_in, n_layer1=n_hidden_1, n_layer2=n_hidden_2, n_out=n_out, learning_rate=lr)

train_loss, val_loss = nn.fit(X, y, max_epochs=num_epochs, get_validation_loss=True)

y_hat_nn = nn.predict_proba(X_test)


def nn_plot(X, y, model):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

    fig, (ax_train, ax_test) = plt.subplots(nrows=1, ncols=2)
    fig.set_figheight(7)
    fig.set_figwidth(15)

    x_min_train, x_max_train = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min_train, y_max_train = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1

    x_min_test, x_max_test = X_test[:, 0].min() - 1, X_test[:, 0].max() + 1
    y_min_test, y_max_test = X_test[:, 1].min() - 1, X_test[:, 1].max() + 1

    step = 0.02
    xx_train, yy_train = np.meshgrid(np.arange(x_min_train, x_max_train, step), np.arange(y_min_train, y_max_train, step))
    Z_train = np.array(model.predict(np.c_[xx_train.ravel(), yy_train.ravel()])).reshape(xx_train.shape)

    xx_test, yy_test = np.meshgrid(np.arange(x_min_test, x_max_test, step), np.arange(y_min_test, y_max_test, step))
    Z_test = np.array(model.predict(np.c_[xx_test.ravel(), yy_test.ravel()])).reshape(xx_test.shape)

    cmap_light = ListedColormap(['#FFAFAA', '#AAFFAA', '#FAAAFF'])
    ax_train.pcolormesh(xx_train, yy_train, Z_train, cmap=cmap_light)
    ax_test.pcolormesh(xx_test, yy_test, Z_test, cmap=cmap_light)

    ax_train.scatter(X_train[np.where(y_train==0), 0], X_train[np.where(y_train==0), 1], label="0")
    ax_train.scatter(X_train[np.where(y_train==1), 0], X_train[np.where(y_train==1), 1], label="1")
    legend = ax_train.legend(title="Classes")

    ax_test.scatter(X_test[np.where(y_test==0), 0], X_test[np.where(y_test==0), 1], label="0")
    ax_test.scatter(X_test[np.where(y_test==1), 0], X_test[np.where(y_test==1), 1], label="1")
    legend = ax_test.legend(title="Classes")

    ax_train.set_xlabel('x1')
    ax_train.set_ylabel('x2')
    ax_train.set_title('Neural Net on Train Data')
    ax_train.set_xlim(xx_train.min(), xx_train.max())
    ax_train.set_ylim(yy_train.min(), yy_train.max())

    ax_test.set_xlabel('x1')
    ax_test.set_ylabel('x2')
    ax_test.set_title('Neural Net on Test Data')
    ax_test.set_xlim(xx_test.min(), xx_test.max())
    ax_test.set_ylim(yy_test.min(), yy_test.max())

    pass
# graphing code found on google
nn_plot(X, y, nn)
