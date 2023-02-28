# TICA

This repository contains a Python implementation of the Topographic ICA algorithm for intended for image decomposition. The original implementation was in MATLAB, and can be found at https://research.ics.aalto.fi/ica/imageica/. 
See (1) and (2) for in depth explainations.

Usage

This Package contains the TICA class, which you first need to instanciate.
TICA(comps = 160, xdim = None, max_iter = 600, verbose = True):

Use .fit() to fit your data to the TICA model. Your data should be a 2D numpy array with the shape [features, samples]
def fit(X, whiten = False, type = 'TICA', dec_type = 'orig', epsi = 0.0005, tol = 1e-4, init_stepsize = 0.1, adapt_n = 5, W_init = None, rand_seed = 1, nb_init = None):
You can pass your own neighborhood matrix and initial vectors as arguments to this function.

After the model has been fit, you can use .view_components() to view the extracted components, and the .mixing_ matrix to decompose your data. 

To generate data like in the original MATLAB package, you can use get_matlab_data(). 
def get_matlab_data(samples = 50000, dataNum = 13, directory = '/path/to/data/', winsize = (16, 16), verbose = True):
The directory should contain dataNum .tiff images numbered 1-dataNum

Unfortunately no proper loss function has been implemented yet, but that is my next step when I take up this project again.

(1) Hyvärinen, A., Hoyer, P. O., & Inki, M. (2001). Topographic independent component analysis. Neural Computation, 13(7), 1527–1558.
(2) Chen, Z., Parvin, D., King, M. & Hao, S. Visualizing Topographic Independent Component Analysis with Movies. Preprint at http://arxiv.org/abs/1901.08239 (2019).
