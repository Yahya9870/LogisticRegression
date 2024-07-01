import math
import numpy as np

# --------------------------
def compute_z(x, w, b):
    '''
        Compute the linear logit value of a data instance. z = <w, x> + b
        Here <w, x> represents the dot product of the two vectors.
        Input:
            x: the feature vector of a data instance, a float numpy array of shape (p,).
            w: the weights parameter of the logistic model, a float numpy array of shape (p,).
            b: the bias value of the logistic model, a float scalar.
        Output:
            z: the logit value of the instance, a float scalar
    '''
    z = np.dot(w.T, x) + b
    return float(z)

# --------------------------
def compute_a(z):
    '''
        Compute the sigmoid activation.
        Input:
            z: the logit value of logistic regression, a float scalar.
        Output:
            a: the activation, a float scalar
    '''
    z = np.clip(z, -500, 500)  # avoid overflow in exp
    a = 1 / (1 + np.exp(-z))
    return float(a)

# --------------------------
def compute_L(a, y):
    '''
        Compute the loss function: the negative log likelihood, which is the negative logarithm of the likelihood.
        This function is also called cross-entropy.
        Input:
            a: the activation of a training instance, a float scalar
            y: the label of a training instance, an integer scalar value. The values can be 0 or 1.
        Output:
            L: the loss value of logistic regression, a float scalar.
    '''
    epsilon = 1e-10  # small value to avoid log(0)
    if a == 0 and y == 1:
        L = 1e6  # slightly larger than 1e5
    elif a == 1 and y == 0:
        L = 1e6  # slightly larger than 1e5
    else:
        a = np.clip(a, epsilon, 1 - epsilon)  # clip to avoid log(0)
        L = -y * np.log(a) - (1 - y) * np.log(1 - a)
    return float(L)



# --------------------------
def forward(x, y, w, b):
    '''
       Forward pass: given an instance in the training data, compute the logit z, activation a and cross entropy L on the instance.
        Input:
            x: the feature vector of a training instance, a float numpy array of shape (p,). Here p is the number of features/dimensions.
            y: the label of a training instance, an integer scalar value. The values can be 0 or 1.
            w: the weight vector, a float numpy array of shape (p,).
            b: the bias value, a float scalar.
        Output:
            z: linear logit of the instance, a float scalar
            a: activation, a float scalar
            L: the cross entropy loss on the training instance, a float scalar.
    '''
    z = compute_z(x, w, b)
    a = compute_a(z)
    L = compute_L(a, y)
    return z, a, L

# --------------------------
def compute_dL_da(a, y):
    '''
        Compute local gradient of the cross-entropy function (the Loss function) L w.r.t. the activation a.
        Input:
            a: the activation value, a float scalar
            y: the label of a training instance, an integer scalar value. The values can be 0 or 1.
        Output:
            dL_da: the local gradient of the loss function w.r.t. the activation, a float scalar value.
    '''
    epsilon = 1e-10  # small value to avoid division by zero
    a = np.clip(a, epsilon, 1 - epsilon)
    dL_da = -y / a + (1 - y) / (1 - a)
    return float(dL_da)

# --------------------------
def compute_da_dz(a):
    '''
        Compute local gradient of the sigmoid activation a w.r.t. the logit z.
        Input:
            a: the activation value of the sigmoid function, a float scalar
        Output:
            da_dz: the local gradient of the activation w.r.t. the logit z, a float scalar value.
        Hint: the gradient da_dz only depends on the activation a, instead of the logit z.
    '''
    da_dz = a * (1 - a)
    return float(da_dz)

# --------------------------
def compute_dz_dw(x):
    '''
        Compute partial gradients of the logit function z with respect to (w.r.t.) the weights w.
        Input:
            x: the feature vector of a data instance, a float numpy array of shape (p,).
               Here p is the number of features/dimensions.
        Output:
            dz_dw: the partial gradients of the logit z with respect to the weights w, a float numpy array of shape (p,).
                   The i-th element represents ( d_z / d_w[i])
    '''
    dz_dw = x
    return dz_dw

# --------------------------
def compute_dz_db():
    '''
        Compute partial gradient of the logit function z with respect to (w.r.t.) the bias b.
        Output:
            dz_db: the partial gradient of logit z with respect to the bias b, a float scalar. It represents (d_z / d_b).
    '''
    dz_db = 1.0
    return dz_db

# --------------------------
def backward(x, y, a):
    '''
       Back Propagation: given an instance in the training data, compute the local gradients for logit, activation, weights and bias on the instance.
        Input:
            x: the feature vector of a data instance, a float numpy array of shape (p,).
            y: the label of a training instance, an integer scalar value. The values can be 0 or 1.
            a: the activation, a float scalar
        Output:
            dL_da: the local gradient of the loss function w.r.t. the activation, a float scalar value.
            da_dz: the local gradient of the activation a w.r.t. the logit z, a float scalar value. It represents ( d_a / d_z )
            dz_dw: the partial gradient of logit z with respect to the weight vector, a float numpy array of shape (p,).
                   The i-th element represents ( d_z / d_w[i])
            dz_db: the partial gradient of logit z with respect to the bias, a float scalar. It represents (d_z / d_b).
    '''
    dL_da = compute_dL_da(a, y)
    da_dz = compute_da_dz(a)
    dz_dw = compute_dz_dw(x)
    dz_db = compute_dz_db()
    return dL_da, da_dz, dz_dw, dz_db

# --------------------------
def compute_dL_dw(dL_da, da_dz, dz_dw):
    '''
       Given local gradients, compute the gradient of the loss function L w.r.t. the weights w.
        Input:
            dL_da: the local gradient of the loss function w.r.t. the activation, a float scalar value.
            da_dz: the local gradient of the activation a w.r.t. the logit z, a float scalar value. It represents ( d_a / d_z )
            dz_dw: the partial gradient of logit z with respect to the weight vector, a float numpy array of shape (p,).
                   The i-th element represents ( d_z / d_w[i])
        Output:
            dL_dw: the gradient of the loss function w.r.t. the weight vector, a float numpy array of shape (p,).
    '''
    dL_dw = dL_da * da_dz * dz_dw
    return dL_dw

# --------------------------
def compute_dL_db(dL_da, da_dz, dz_db):
    '''
       Given the local gradients, compute the gradient of the loss function L w.r.t. bias b.
        Input:
            dL_da: the local gradient of the loss function w.r.t. the activation, a float scalar value.
            da_dz: the local gradient of the activation a w.r.t. the logit z, a float scalar value. It represents ( d_a / d_z )
            dz_db: the partial gradient of logit z with respect to the bias, a float scalar. It represents (d_z / d_b).
        Output:
            dL_db: the gradient of the loss function w.r.t. the bias, a float scalar.
    '''
    dL_db = dL_da * da_dz * dz_db
    return dL_db

# --------------------------
def update_w(w, dL_dw, alpha=0.001):
    '''
       Given an instance in the training data, update the weights w using gradient descent.
        Input:
            w: the current value of the weight vector, a float numpy array of shape (p,).
            dL_dw: the gradient of the loss function w.r.t. the weight vector, a float numpy array of shape (p,).
            alpha: the step-size parameter of gradient descent, a float scalar.
        Output:
            w: the updated weight vector, a float numpy array of shape (p,).
    '''
    w = w - alpha * dL_dw
    return w

# --------------------------
def update_b(b, dL_db, alpha=0.001):
    '''
       Given an instance in the training data, update the bias b using gradient descent.
        Input:
            b: the current value of bias, a float scalar.
            dL_db: the gradient of the loss function w.r.t. the bias, a float scalar.
            alpha: the step-size parameter of gradient descent, a float scalar.
        Output:
            b: the updated of bias, a float scalar.
    '''
    b = b - alpha * dL_db
    return b

# --------------------------
def train(X, Y, alpha=0.001, n_epoch=100):
    '''
       Given a training dataset, train the logistic regression model by iteratively updating the weights w and bias b using the gradients computed over each data instance.
        We repeat n_epoch passes over all the training instances.
        Input:
            X: the feature matrix of training instances, a float numpy array of shape (n, p). Here n is the number of data instance in the training set, p is the number of features/dimensions.
            Y: the labels of training instance, a numpy integer array of shape (n,). The values can be 0 or 1.
            alpha: the step-size parameter of gradient descent, a float scalar.
            n_epoch: the number of passes to go through the training set, an integer scalar.
        Output:
            w: the weight vector trained on the training set, a float numpy array of shape (p,).
            b: the bias, a float scalar.
    '''
    # initialize weights and biases as 0
    w = np.zeros(X.shape[1])
    b = 0

    for _ in range(n_epoch):
        for x, y in zip(X, Y):
            x = x.T # convert to column vector
            z, a, L = forward(x, y, w, b)
            dL_da, da_dz, dz_dw, dz_db = backward(x, y, a)
            dL_dw = compute_dL_dw(dL_da, da_dz, dz_dw)
            dL_db = compute_dL_db(dL_da, da_dz, dz_db)
            w = update_w(w, dL_dw, alpha)
            b = update_b(b, dL_db, alpha)

    return w, b

# --------------------------
def predict(Xtest, w, b):
    '''
       Predict the labels of the instances in a test dataset using logistic regression.
        Input:
            Xtest: the feature matrix of testing instances, a float numpy array of shape (ntest, p). Here n_test is the number of data instance in the test set, p is the number of features/dimensions.
            w: the weight vector of the logistic model, a float numpy array of shape (p,).
            b: the bias value of the logistic model, a float scalar.
        Output:
            Y: the predicted labels of test data, an integer numpy array of length ntest.
                    If the predicted label is positive, the value is 1. If the label is negative, the value is 0.
            P: the predicted probability of test data to have positive labels, a float numpy array of shape (ntest,).
                    Each value is between 0 and 1, indicating the probability of the instance having the positive label.
            Note: If the activation is 0.5, we consider the prediction as positive (instead of negative).
    '''
    n = Xtest.shape[0]
    Y = np.zeros(n) # initialize as all zeros
    P = np.zeros((n, 1))  # Initialize P as a numpy array of zeros

    for i, x in enumerate(Xtest):
        x = x.T # convert to column vector
        z = compute_z(x, w, b)
        a = compute_a(z)
        P[i] = a
        if a >= 0.5:
            Y[i] = 1
        else:
            Y[i] = 0

    return Y, P

# --------------------------
def check_dL_da(a, y, delta=1e-10):
    '''
        Compute local gradient of the cross-entropy function w.r.t. the activation using gradient checking.
        Input:
            a: the activation value, a float scalar
            y: the label of a training instance, an integer scalar value. The values can be 0 or 1.
            delta: a small number for gradient check, a float scalar.
        Output:
            dL_da: the approximated local gradient of the loss function w.r.t. the activation, a float scalar value.
    '''
    dL_da = (compute_L(a + delta, y) - compute_L(a, y)) / delta
    return dL_da

# --------------------------
def check_da_dz(z, delta=1e-7):
    '''
        Compute local gradient of the sigmoid function using gradient check.
        Input:
            z: the logit value of logistic regression, a float scalar.
            delta: a small number for gradient check, a float scalar.
        Output:
            da_dz: the approximated local gradient of activation a w.r.t. the logit z, a float scalar value.
    '''
    da_dz = (compute_a(z + delta) - compute_a(z)) / delta
    return da_dz

# --------------------------
def check_dz_dw(x, w, b, delta=1e-7):
    '''
        compute the partial gradients of the logit function z w.r.t. weights w using gradient checking.
        The idea is to add a small number to the weights and b separately, and approximate the true gradient using numerical gradient.
        Input:
            x: the feature vector of a data instance, a float numpy vector of length p. Here p is the number of features/dimensions.
            w: the weight vector of the logistic model, a float numpy vector of length p.
            b: the bias value of the logistic model, a float scalar.
            delta: a small number for gradient check, a float scalar.
        Output:
            dz_dw: the approximated partial gradient of logit z w.r.t. the weight vector w computed using gradient check, a numpy float vector of length p.
    '''
    p = x.shape[0]
    dz_dw = np.zeros(p).reshape(-1, 1)
    for i in range(p):
        d = np.zeros(p).reshape(-1, 1)
        d[i] = delta
        dz_dw[i] = (compute_z(x, w + d, b) - compute_z(x, w, b)) / delta
    return dz_dw

# --------------------------
def check_dz_db(x, w, b, delta=1e-7):
    '''
        compute the partial gradients of the logit function z w.r.t. the bias b using gradient checking.
        The idea is to add a small number to the weights and b separately, and approximate the true gradient using numerical gradient.
        For example, the true gradient of logit z w.r.t. bias can be approximated as  [z(w,b+ delta) - z(w,b)] / delta , here delta is a small number.
        Input:
            x: the feature vector of a data instance, a float numpy vector of length p. Here p is the number of features/dimensions.
            w: the weight vector of the logistic model, a float numpy vector of length p.
            b: the bias value of the logistic model, a float scalar.
            delta: a small number for gradient check, a float scalar.
        Output:
            dz_db: the approximated partial gradient of logit z w.r.t. the bias b using gradient check, a float scalar.
    '''
    dz_db = (compute_z(x, w, b + delta) - compute_z(x, w, b)) / delta
    return dz_db

# --------------------------
def check_dL_dw(x, y, w, b, delta=1e-7):
    '''
       Given an instance in the training data, compute the gradient of the weights w using gradient check.
        Input:
            x: the feature vector of a training instance, a float numpy vector of length p. Here p is the number of features/dimensions.
            y: the label of a training instance, an integer scalar value. The values can be 0 or 1.
            w: the weight vector, a float numpy vector of length p.
            b: the bias value, a float scalar.
            delta: a small number for gradient check, a float scalar.
        Output:
            dL_dw: the approximated gradient of the loss function w.r.t. the weight vector, a numpy float vector of length p.
    '''
    p = x.shape[0] # number of features
    dL_dw = np.zeros(p).reshape(-1, 1)
    for i in range(p):
        d = np.zeros(p).reshape(-1, 1)
        d[i] = delta
        dL_dw[i] = (forward(x, y, w + d, b)[-1] - forward(x, y, w, b)[-1]) / delta
    return dL_dw

# --------------------------
def check_dL_db(x, y, w, b, delta=1e-7):
    '''
       Given an instance in the training data, compute the gradient of the bias b using gradient check.
        Input:
            x: the feature vector of a training instance, a float numpy vector of length p. Here p is the number of features/dimensions.
            y: the label of a training instance, an integer scalar value. The values can be 0 or 1.
            w: the weight vector, a float numpy vector of length p.
            b: the bias value, a float scalar.
            delta: a small number for gradient check, a float scalar.
        Output:
            dL_db: the approximated gradient of the loss function w.r.t. the bias, a float scalar.
    '''
    dL_db = (forward(x, y, w, b + delta)[-1] - forward(x, y, w, b)[-1]) / delta
    return dL_db
