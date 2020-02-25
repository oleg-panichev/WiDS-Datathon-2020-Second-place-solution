import datetime
import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


def dprint(*args, **kwargs):
    print("[{}] ".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")) + \
        " ".join(map(str,args)), **kwargs)


class LRSGD(BaseEstimator):
    """ A template estimator to be used as a reference implementation.
    For more information regarding how to build your own estimator, read more
    in the :ref:`User Guide <user_guide>`.
    Parameters
    ----------
    lr : float, default=0.01
        Learning rate.
    """
    def __init__(self, objective='regression', lr=0.01, epochs=1000, 
                 weight_decay=0,
                 reduce_lr=False, reduce_lr_factor=0.1, reduce_lr_epochs=100,
                 early_stopping=True, early_stopping_epochs=500,
                 eps=1e-6, print_step=0, verbose=False):

        if objective not in ['regression', 'binary_classification']:
            return ValueError
        self.objective = objective
        if self.objective == 'regression':
            self.cost_fn = self.__mse_loss
        elif self.objective == 'binary_classification':
            self.cost_fn = self.__binary_crossentropy
        self.lr = lr
        self.epochs = epochs
        self.weight_decay = weight_decay
        self.reduce_lr = reduce_lr 
        self.reduce_lr_factor = reduce_lr_factor 
        self.reduce_lr_epochs = reduce_lr_epochs
        self.early_stopping = early_stopping
        self.early_stopping_epochs = early_stopping_epochs
        self.eps = eps
        self.print_step = print_step
        self.verbose = verbose

    def __log_softmax(self, x):
        return x - x.exp().sum(-1).log().unsqueeze(-1)

    def __mse_loss(self, X, theta, y, n):
        cost = (1/(2*n))*np.transpose((X@theta - y))@(X@theta - y)
        derv = (1/n)*np.transpose(X)@(X@theta - y)
        return cost, derv

    def __binary_crossentropy(self, X, theta, y, n, eps=1e-6):
        cost = -(1/n) * np.sum(y * np.log(self.__sigmoid(X@theta) + eps) + (1 - y) * np.log(1 - self.__sigmoid(X@theta) + eps))
        derv = (1/n) * np.dot(X.T, self.__sigmoid(X@theta) - y)
        return cost, derv

    def __sigmoid(self, x):
        # Activation function used to map any real value between 0 and 1
        return 1 / (1 + np.exp(-x))

    def fit(self, X, y, validation=None):
        """A reference implementation of a fitting function.
        Parameters
        ----------
        X : {array-like}, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression).
        Returns
        -------
        self : object
            Returns self.
        """
        X, y = check_X_y(X, y, accept_sparse=False)
        self.is_fitted_ = True
        
        n_features = X.shape[1]
        X = np.array(X)
        X = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)
        n = float(len(X))
#         X = np.concatenate([np.ones((X.shape[0], 1)), X, X], axis=1)
        theta = np.random.randn(X.shape[1])
        # theta = np.ones(X.shape[1])
#         theta[-n_features:] = 1

        if validation is not None:
            (X_valid, y_valid) = validation
            X_valid = np.array(X_valid)
            X_valid = np.concatenate([np.ones((X_valid.shape[0], 1)), X_valid], axis=1)
            n_valid = float(len(X_valid))

        best_cost = np.nan

        reduce_lr_cnt = 0
        early_stopping_cnt = 0

        derv = 0
        cost = 0
        for i in range(self.epochs): 
            y_pred = X@theta
            cost, derv = self.cost_fn(X, theta, y, n)

            if validation is not None:
                cost_valid, _ = self.cost_fn(X_valid, theta, y_valid, n_valid)
                cost_mntr = cost_valid
            else:
                cost_mntr = cost

            if i == 0:
                best_cost = cost_mntr

            theta = theta - self.weight_decay*self.lr*theta - self.lr*derv

            if cost_mntr < best_cost and np.abs(cost_mntr - best_cost) > self.eps:
                reduce_lr_cnt = 0
                early_stopping_cnt = 0
                best_cost = cost_mntr
            else:
                reduce_lr_cnt += 1
                early_stopping_cnt += 1

            if self.verbose and (self.print_step > 0 and (i%self.print_step == 0 or i == self.epochs - 1)):
                if validation is not None:
                    dprint(f'[Epoch {i:5d}] loss: {cost:.8f} loss valid: {cost_valid:.8f} best loss: {best_cost:.8f}')
                else:
                    dprint(f'[Epoch {i:5d}] loss: {cost:.8f} best loss: {best_cost:.8f}')

            if self.reduce_lr:
                if reduce_lr_cnt >= self.reduce_lr_epochs:
                    self.lr = self.lr*self.reduce_lr_factor
                    reduce_lr_cnt = 0

                    if self.verbose:
                        dprint(f'[Epoch {i:5d}] Reducing LR to {self.lr}.')


            if self.early_stopping: 
                if early_stopping_cnt >= self.early_stopping_epochs:
                    if self.verbose:
                        if validation is not None:
                            dprint(f'[Epoch {i:5d}] loss: {cost:.8f} loss valid: {cost_valid:.8f} best loss: {best_cost:.8f}')
                        else:
                            dprint(f'[Epoch {i:5d}] loss: {cost:.8f} best loss: {best_cost:.8f}')
                        dprint(f'Early stopping on epoch {i}.')
                    break
                
            cost_prev = cost

        self.theta = theta
    
        return self

    def predict(self, X):
        """ A reference implementation of a predicting function.
        Parameters
        ----------
        X : {array-like}, shape (n_samples, n_features)
            The training input samples.
        Returns
        -------
        y : ndarray, shape (n_samples,)
            Returns an array of ones.
        """
        X = check_array(X, accept_sparse=True)
        check_is_fitted(self, 'is_fitted_')
        X = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)

        if self.objective == 'regression':
            return X@self.theta
        elif self.objective == 'binary_classification':
            return self.__sigmoid(X@self.theta)



