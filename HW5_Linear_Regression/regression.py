import numpy as np
from matplotlib import pyplot as plt
import csv


# Feel free to import other packages, if needed.
# As long as they are supported by CSL machines.


def get_dataset(filename):
    """
    TODO: implement this function.

    INPUT: 
        filename - a string representing the path to the csv file.

    RETURNS:
        An n by m+1 array, where n is # data points and m is # features.
        The labels y should be in the first column.
    """
    dataset = []
    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)            # skip first row
        for row in reader:
            temp = []
            for data in row[1:]:
                temp.append(float(data))        # read values as floats
            dataset.append(temp)
    return np.array(dataset)


def print_stats(dataset, col):
    """
    TODO: implement this function.

    INPUT: 
        dataset - the body fat n by m+1 array
        col     - the index of feature to summarize on. 
                  For example, 1 refers to density.

    RETURNS:
        None
    """
    n = len(dataset)
    print(n)            # print the number of data points
    mean = 0
    for data in dataset[:, col]:
        mean += data
    mean = mean / n         # print the mean
    print('{:.2f}'.format(float(mean)))
    std = 0
    for data in dataset[:, col]:
        std += (data - mean) ** 2
    std = std / (n - 1)
    std = np.sqrt(std)
    print('{:.2f}'.format(float(std)))      # print the SD


def regression(dataset, cols, betas):
    """
    TODO: implement this function.

    INPUT: 
        dataset - the body fat n by m+1 array
        cols    - a list of feature indices to learn.
                  For example, [1,8] refers to density and abdomen.
        betas   - a list of elements chosen from [beta0, beta1, ..., betam]

    RETURNS:
        mse of the regression model
    """
    mse = 0
    n = len(dataset)
    for i in range(0, n):           # do the sum from i = 1 through n
        temp = betas[0] - dataset[i][0]
        for j in range(0, len(cols)):
            temp += betas[j + 1] * dataset[i][cols[j]]
        mse += temp ** 2
    return mse / n


def gradient_descent(dataset, cols, betas):
    """
    TODO: implement this function.

    INPUT: 
        dataset - the body fat n by m+1 array
        cols    - a list of feature indices to learn.
                  For example, [1,8] refers to density and abdomen.
        betas   - a list of elements chosen from [beta0, beta1, ..., betam]

    RETURNS:
        An 1D array of gradients
    """
    grads = []
    n = len(dataset)
    for i in range(0, len(betas)):      # find gradient for each beta
        temp_sum = 0
        for j in range(0, n):           # do the sum from i = 1 through n
            temp = betas[0] - dataset[j][0]
            for k in range(0, len(cols)):
                temp += betas[k + 1] * dataset[j][cols[k]]
            if i > 0:               # skip for beta0
                temp = temp * dataset[j][cols[i - 1]]
            temp_sum += temp
        grads.append(temp_sum * 2 / n)
    return np.array(grads)


def iterate_gradient(dataset, cols, betas, T, eta):
    """
    TODO: implement this function.

    INPUT: 
        dataset - the body fat n by m+1 array
        cols    - a list of feature indices to learn.
                  For example, [1,8] refers to density and abdomen.
        betas   - a list of elements chosen from [beta0, beta1, ..., betam]
        T       - # iterations to run
        eta     - learning rate

    RETURNS:
        None
    """
    for i in range(1, T + 1):
        grads = gradient_descent(dataset, cols, betas)      # grads for betas^(t-1)
        for j in range(0, len(betas)):
            betas[j] = betas[j] - eta * grads[j]            # get betas^(t) from betas^(t-1)
        print(i, '{:.2f}'.format(regression(dataset, cols, betas)), end=" ")        # print iteration num and mse
        # print betas, only change line at the end of every iteration
        for j in range(0, len(betas) - 1):
            print('{:.2f}'.format(betas[j]), end=" ")
        print('{:.2f}'.format(betas[len(betas) - 1]))


def compute_betas(dataset, cols):
    """
    TODO: implement this function.

    INPUT: 
        dataset - the body fat n by m+1 array
        cols    - a list of feature indices to learn.
                  For example, [1,8] refers to density and abdomen.

    RETURNS:
        A tuple containing corresponding mse and several learned betas
    """
    X_T = [list(np.ones(len(dataset)))]         # insert a row of ones to the beginning of X_T
    # put data in to X_T, it should be len(cols)+1 by len(dataset) after this loop
    for col in cols:
        X_T.append(dataset[:, col])
    X = list(np.transpose(X_T))         # get X from X transpose
    betas = np.dot(np.dot(np.linalg.inv(np.dot(X_T, X)), X_T), dataset[:, 0])       # formula to get betas
    mse = regression(dataset, cols, betas)          # get mse using calculated betas
    return (mse, *betas)


def predict(dataset, cols, features):
    """
    TODO: implement this function.

    INPUT: 
        dataset - the body fat n by m+1 array
        cols    - a list of feature indices to learn.
                  For example, [1,8] refers to density and abdomen.
        features- a list of observed values

    RETURNS:
        The predicted body fat percentage value
    """
    betas = np.array(list(compute_betas(dataset, cols))[1:])        # ignore mse returned from compute_betas
    result = betas[0] + np.dot(betas[1:], np.transpose(np.array(features)))         # calculate yi
    return result


def synthetic_datasets(betas, alphas, X, sigma):
    """
    TODO: implement this function.

    Input:
        betas  - parameters of the linear model
        alphas - parameters of the quadratic model
        X      - the input array (shape is guaranteed to be (n,1))
        sigma  - standard deviation of noise

    RETURNS:
        Two datasets of shape (n,2) - linear one first, followed by quadratic.
    """
    z = np.random.normal(0, sigma, len(X))          # noise
    linear_y = []
    quadratic_y = []
    for i in range(0, len(X)):
        linear_y.append(betas[0] + betas[1] * X[i] + z[i])         # linear dataset
        quadratic_y.append(alphas[0] + alphas[1] * (X[i] ** 2) + z[i])       # quadratic dataset
    return np.append(linear_y, X, axis=1), np.append(quadratic_y, X, axis=1)        # y in left col, x in right col


def plot_mse():
    from sys import argv
    if len(argv) == 2 and argv[1] == 'csl':
        import matplotlib
        matplotlib.use('Agg')

    # TODO: Generate datasets and plot an MSE-sigma graph
    X = np.transpose([np.random.uniform(-100, 100, 1000)])
    betas = np.array([1, 2])
    alphas = np.array([1, 1])
    sigmas = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]
    mse1 = []           # for linear dataset
    mse2 = []           # for quadratic dataset
    for sigma in sigmas:
        dataset1, dataset2 = synthetic_datasets(betas, alphas, X, sigma)
        mse1.append(compute_betas(dataset1, [1])[0])
        mse2.append(compute_betas(dataset2, [1])[0])
    plt.plot(sigmas, mse1, marker="o")
    plt.plot(sigmas, mse2, marker="o")
    plt.legend(("MSE of Linear Dataset", "MSE of Quadratic Dataset"))
    plt.yscale('log')
    plt.ylabel('MSE of Trained Model')
    plt.xscale('log')
    plt.xlabel('Standard Deviation of Error Term')
    plt.savefig('mse.pdf')


if __name__ == '__main__':
    ### DO NOT CHANGE THIS SECTION ###
    plot_mse()
