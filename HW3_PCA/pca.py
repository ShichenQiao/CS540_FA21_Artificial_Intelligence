from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt


def load_and_center_dataset(filename):
    # TODO: add your code here
    x = np.load(filename)       # load the dataset from a provided .npy file
    x_centered = x - np.mean(x, axis=0)     # re-center it around the origin
    return x_centered       # return it as a NumPy array of floats


def get_covariance(dataset):
    # TODO: add your code here
    # calculate the summation first
    S = np.dot(np.transpose([dataset[0]]), [dataset[0]])
    for i in range(1, len(dataset)):
        S += np.dot(np.transpose([dataset[i]]), [dataset[i]])
    S = S / (len(dataset) - 1)      # then divide by n - 1 to get the covariance matrix
    return S        # d * d matrix


def get_eig(S, m):
    # TODO: add your code here
    # get m Largest Eigenvalues/Eigenvectors in ascending order
    eig_val, eig_vec = eigh(S, eigvals=(len(S) - m, len(S) - 1))
    # rearrange the output of eigh() to get the eigenvalues in decreasing order
    eig_val = np.flip(eig_val)
    eig_vec = np.flip(eig_vec, axis=1)
    # return the diagonal matrix of largest m eigenvalues FIRST, then the eigenvectors in corresponding columns
    return np.diag(eig_val), eig_vec


def get_eig_perc(S, perc):
    # TODO: add your code here
    # get all Eigenvalues/Eigenvectors in ascending order
    eig_val, eig_vec = eigh(S, eigvals=(0, len(S) - 1))
    # rearrange the output of eigh() to get the eigenvalues in decreasing order
    eig_val = np.flip(eig_val)
    eig_vec = np.flip(eig_vec, axis=1)
    # find the cutoff index of the eigenvalues that explain more than the given percentage of variance
    cutoff_index = 0
    while True:
        if eig_val[cutoff_index] / np.sum(eig_val) <= perc:
            break
        cutoff_index += 1
    # return the diagonal matrix of satisfied eigenvalues FIRST, then the eigenvectors in corresponding columns
    return np.diag(eig_val[0:cutoff_index]), eig_vec[:, 0:cutoff_index]


def project_image(img, U):
    # TODO: add your code here
    m = len(U[0])       # get the dimension m from U
    projection = np.zeros(shape=(1, len(img)))      # define the returning projection vector
    # project the given image into the m dimensional space
    for j in range(0, m):
        alpha = np.dot([U[:, j]], np.transpose([img]))      # calculate alpha
        projection += alpha * [U[:, j]]     # add up to the projection vector
    return projection


def display_image(orig, proj):
    # TODO: add your code here
    # reshape the images to be 32x32
    orig = np.transpose(np.reshape([orig], newshape=(32, 32)))
    proj = np.transpose(np.reshape(proj, newshape=(32, 32)))
    # create a figure with one row of two subplots
    fig, axs = plt.subplots(1, 2)
    # set the titles of the subplots
    axs[0].set_title("Original")
    axs[1].set_title("Projection")
    # render the original image in the first subplot and the projection in the second subplot
    orig_img = axs[0].imshow(orig, aspect='equal')
    proj_img = axs[1].imshow(proj, aspect='equal')
    # create a colorbar (Links to an external site.) for each image
    fig.colorbar(orig_img, ax=axs[0])
    fig.colorbar(proj_img, ax=axs[1])
    # render the plot
    plt.show()
