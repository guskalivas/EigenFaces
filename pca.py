# Name: Gus Kalivas
# Program: p5 pca
# wisc: wkalivas

from scipy.io import loadmat
from scipy.linalg import eigh
import scipy
import numpy as np
import matplotlib
from matplotlib import pyplot, transforms

'''
load and center dataset takes in a filename and creates a numpy array from the'fea' data
then calculates the average and subtracts the dataset from the average to center data
and returns centered matrix
'''
def load_and_center_dataset(filename):
    dataset = loadmat(filename) #loads data from .mat file
    x = np.array(dataset['fea'], float)
    mean = np.mean(x, axis= 0) # calculates mean
    sub = x - mean
    return sub

x = load_and_center_dataset('YaleB_32x32.mat')
#print(len(x), len(x[0]), np.average(x))

'''
get covariance calculates the covariance for the dataset matrix
and returns a 1024X1024 dataset
'''
def get_covariance(dataset):
    x = np.transpose(dataset) # transpose dataset
    d = np.dot(x, dataset) # take the dot product of the two
    yy = d*(1/(len(dataset) -1)) # multiple by 1/n-1
    return yy

c = get_covariance(x)
#print(len(c), len(c[0]))


'''
get_eig takes in the covariance matrix and number of values to display
'''
def get_eig(S, m):
    Lambda, U = scipy.linalg.eigh(S, eigvals = (len(S)-m, len(S)-1)) # get highest m values
    c = np.fliplr(U) # flip the matrix
    vals = Lambda[::-1] # sort the values 
    vals = np.diag(Lambda, k = 0) # create diag matrix 
    return vals, c # return vector and values

Lambda, U = get_eig(c, 2)

'''
project image takes in an image vectors and eigh vectors 
'''
def project_image(image, U):
    dot = np.dot(image, U) # dot product of the two 
    d2 = np.dot(dot, np.transpose(U)) # dot product of U transposed and first dot product
    return d2

p = project_image(x[4], U)
#print(p)

'''
Takes in the original image and the project image and creates two pictures of the two
'''
def display_image(orig, proj):
    # create two subplots in one figure 
    f, (ax1, ax2) = matplotlib.pyplot.subplots(nrows = 1, ncols= 2)
    # reshape both of the images 
    orig = np.reshape(orig, newshape = (32,32))
    proj = np.reshape(proj, newshape = (32,32))
    # set the titles 
    ax1.set_title('Original')
    ax2.set_title('Projection')
    # show each image and transpose the data to flip correct way
    pos = ax1.imshow(np.transpose(orig), aspect='equal')
    sec = ax2.imshow(np.transpose(proj), aspect='equal')
    # create a color bar with the two images and correct spacing 
    f.colorbar(pos, ax=ax1, fraction=0.046, pad=0.04)
    f.colorbar(sec, ax= ax2, fraction=0.046, pad=0.04)
    return matplotlib.pyplot.show() 

display_image(x[35], p)
