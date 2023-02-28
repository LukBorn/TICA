import numpy as np
from scipy import linalg
import cv2
import matplotlib.pyplot as plt

def _decorrelation(W, type = 'orig'):
    """
    :param W:
        vectors to decorrelate
    :param type
        'orig': adapted from the original MATLAB Package
        'scipy': scipy decorrelation
        'sym': symmetric decorrelation adapted from sklearn FastICA
    """
    if type == 'orig':
        # B = B*real((B'*B)^(-0.5));
        return np.dot(W, np.real(linalg.fractional_matrix_power(np.dot(W.T, W), -0.5)))
    elif type == 'scipy':
        return linalg.orth(W)
    elif type == 'sym':
        s, u = linalg.eigh(np.dot(W, W.conj().T))
        s = np.clip(s, a_min=np.finfo(W.dtype).tiny, a_max=None)
        return np.linalg.multi_dot([u * (1.0 / np.sqrt(s)), u.conj().T, W])

class TICA:
    """
    Class for performing topographic independent component analysis for image decomposition
    Hyvärinen, A., Hoyer, P. O., & Inki, M. (2001). Topographic independent component analysis.
        Neural Computation, 13(7), 1527–1558.

    Adapted from MATLAB Package available at https://research.ics.aalto.fi/ica/imageica/

    Attributes:
        comps:
            number of components
        xdim:
            number of xdimensions of the topography matrix
        ydim:
            number of ydimensions of the topography matrix
            xdim*ydim must equal components
        max_iter:
            maximum number of iterations allowed for gradient
            since there is currently no loss function implemented,
            this is the only control for the iteration.
            For TICA this should be at least 600, for fastICA at least 100
        nbmaptype:
            type of map for the neighborhood matrix, ['torus, 'standard]
        stepsize:
            initial stepsize for the gradient
        verbose:
            prints "progress reports" for the calculation if True

    function .fit():
        fits the model to data in shape [features, samples]

    Attributes after fitting:
        components_:
            the linear operator to apply to the data to get independent sources
        mixing_:
            the linear operator that maps independet sources to the data
        mean_:
            mean over features, needs to be set before fitting if whiten is False
        whitening:
            whitening Matrix, needs to be set before fitting if whiten is False
        dewhitening_:
            dewhitening Matrix, needs to be set before fitting if whiten is False


    """
    def __init__(self,
                 comps = 160,
                 xdim = None,
                 max_iter = 600,
                 verbose = True):

        if xdim == None:
            xdim = int(np.sqrt(comps))
            ydim = int(comps / int(np.sqrt(comps)))
        else:
            ydim = int(comps/xdim)
        if xdim*ydim != comps:
            print(f'WARNING: xdim ({xdim}) * ydim ({ydim}) does not equal components ({comps})\n'
                  f'adjusting components to {xdim*ydim}')
        self._xdim = xdim
        self._ydim = ydim
        self.n_components = xdim*ydim

        self.max_iter = max_iter
        self.verbose = verbose

        self.nbmaptype = 'torus'
        self.stepsize = 0.1

    def fit(self, X,
            whiten = False, type = 'TICA', dec_type = 'orig',
            epsi = 0.0005, tol = 1e-4,
            init_stepsize = 0.1, adapt_n = 5, W_init = None):
        """
        :param X:
            the data to extract the components from:
            2D numpy array of shape [features, samples]
            for images or image tiles just flatten the images
        :param type:
            ['fastICA', 'TICA'
        :param dec_type:
            type of decorrelation
            ['sym', 'orig', 'scipy']
        :param whiten:
            True if data is prewhitened
        :param epsi:
            small constant for numeric stability of the nonlinearity g
        :param tol:
            tolerance factor for learning rule, relates to adapted stepsize
        :param init_stepsize:
            initial stepsize of the algorithm (recommended 0.1)
        :param W_init:
            initial component vectors
            2D numpy array of shape [components, components]
            if None are given, random vectors are generated from gaussian distribution
        """
        if not whiten:
            # Subtract the mean of each row of X
            if self.verbose:
                print('Subtracting local mean...')
            self.mean_ = np.mean(X, axis=1, keepdims=True)
            X = X - self.mean_

            # Compute the covariance matrix
            if self.verbose:
                print('Calculating covariance...')
            covarianceMatrix = X @ X.T / X.shape[1]

            # Compute the eigendecomposition of the covariance matrix
            eigenvalues, eigenvectors = np.linalg.eig(covarianceMatrix)

            # Sort the eigenvalues and select the top rdim eigenvectors
            if self.verbose:
                print('Reducing dimensionality and whitening...')
            order = np.argsort(-eigenvalues)
            E = eigenvectors[:, order[:self.n_components]]
            d = np.real(eigenvalues[order[:self.n_components]] ** (-0.5))
            D = np.diag(d)

            # Whiten the data by projecting onto the rdim eigenvectors
            X = D @ E.T @ X

            # Compute the whitening and dewhitening matrices
            self.whitening_ = D @ E.T
            self.dewhitening_ = E @ np.linalg.inv(D)


        nb = Neighborhood(self._xdim, self._ydim, maptype=self.nbmaptype, verbose = self.verbose)

        if W_init is not None:
            W = W_init
        else:
            if self.verbose:
                print('Generating random initial vectors...')
            # random initial vectors
            w_init = np.asarray(np.random.normal(size=(self.n_components, self.n_components)), dtype=X.dtype)
            # definitely correct
            W = _decorrelation(w_init, dec_type)

        stepsize = init_stepsize

        if type == "TICA":
            if self.verbose:
                print(f'Initiatizing topographic ICA iteration with stepsize {stepsize}...')
            try:
                for ii in range(self.max_iter):
                    if self.verbose:
                        print(f'({ii}), ', end='')

                    # calculate linear filter responses and their squares
                    x = np.dot(W.conj().T, X)
                    xsq = x ** 2

                    # calculate local energies
                    E = np.zeros(shape=(nb.NBNZ.shape[0], X.shape[1]))
                    for i in range(1, W.shape[1]):
                        E[i] = np.dot(nb.NB[i, nb.NBNZ[i]], xsq[nb.NBNZ[i], :])

                    # take nonlinearity
                    g = -((E + epsi) ** -0.5)

                    # calculate convolution with neighborhood
                    F = np.zeros(shape=(nb.NBNZ.shape[0], X.shape[1]))
                    for i in range(1, W.shape[1]):
                        F[i] = np.dot(nb.NB[i, nb.NBNZ[i]], g[nb.NBNZ[i], :])

                    # total gradient
                    deltaB = np.dot(X, (np.multiply(x, F)).conj().T) / X.shape[1]

                    # adapt stepsize only every fifth time
                    if ii % adapt_n == 0:
                        adapt_ls = [0.5, 1, 2]
                        objective_ls = []
                        for i in adapt_ls:
                            W_adapt = _decorrelation(W + np.multiply(i * stepsize, deltaB), dec_type)
                            xsq = (np.dot(W_adapt.conj().T, X))**2
                            # calculate local energies
                            E = np.zeros(shape=(nb.NBNZ.shape[0], X.shape[1]))
                            for i in range(1, W.shape[1]):
                                E[i] = np.dot(nb.NB[i, nb.NBNZ[i]], xsq[nb.NBNZ[i], :])
                            objective = np.mean(np.mean(np.sqrt(epsi + E), axis=0))
                            objective_ls.append(objective)
                            del W_adapt, xsq, E
                        stepsize *= adapt_ls[objective_ls.index(min(objective_ls))]
                        if self.verbose:
                            print(f' Adapting stepsize to: {stepsize}, '
                                  f'Objective value: {min(objective_ls)}')

                    W1 = W + stepsize * deltaB
                    W = _decorrelation(W1, dec_type)

                    #TODO: implement a proper loss function
                    if stepsize < tol:
                        if self.verbose:
                            print(f'Iteration converged in {ii} iterations')
                        break

                else:
                    print("\nWARNING: Iteration didn't converge. Allow more iterations or a higher tolerance")
            except KeyboardInterrupt:
                print(f'\nIteration interrupted at iteration {ii}')

        elif type == "fastICA":
            if self.verbose:
                print(f'Initiatizing fastICA iteration...')
            for ii in range(self.max_iter):
                if self.verbose:
                    print(f'({ii}), ', end=''if ii%10 == 0 else '\n')
                hyptan = np.tanh(np.dot(X.T, W))
                mean_hyp_tan_sq = np.mean(1 - hyptan ** 2, axis=0, keepdims=True)
                W1 = np.dot(X, hyptan/X.shape[1]) - mean_hyp_tan_sq * W
                W = _decorrelation(W1, dec_type)

        self.components_ = np.dot(self.dewhitening_, W)
        self.mixing_ = np.dot(W.T, self.whitening_)

    def view_components(self, tiledims):
        """
        plots components in their topography

        :param tiledims:
            tuple containing the dimensions of the image patches
        """
        components_ = self.components_.T.reshape((self.n_components, tiledims[0], tiledims[1]))
        fig, ax = plt.subplots(self._ydim, self._xdim)
        for i, a in enumerate(ax.flatten()):
            a.imshow(components_[i,:,:], cmap='gray')
            a.set_xticks([])
            a.set_yticks([])
            a.spines['top'].set_visible(False)
            a.spines['bottom'].set_visible(False)
            a.spines['left'].set_visible(False)
            a.spines['right'].set_visible(False)

class Neighborhood:
    def __init__(self,
                 xdim,
                 ydim,
                 maptype = 'torus',
                 nbshape= 'square',
                 nbsize = 3,
                 verbose = True
                ):

        assert maptype in ['torus','standard'], f"maptype not in 'torus' or 'standard'"
        self.maptype = maptype
        assert nbshape in ['square'], 'only square neighborhood shape implemented'
        self.nbshape = nbshape
        self.nbsize = nbsize

        self.xdim = xdim
        self.ydim = ydim


        NB = np.zeros((self.xdim * self.ydim, self.xdim * self.ydim))

        if self.nbshape == 'square':
            size = int((self.nbsize - 1) / 2)
            ind = 0
            for y in range(1, self.ydim + 1):
                for x in range(1, self.xdim + 1):
                    ind += 1

                    # Rectangular neighbors
                    xn, yn = np.meshgrid(np.arange(x - size, x + size + 1), np.arange(y - size, y + size + 1))
                    xn = xn.reshape(-1)
                    yn = yn.reshape(-1)

                    if self.maptype == 'torus':

                        # Cycle round
                        yn[yn < 1] += self.ydim
                        yn[yn > self.ydim] -= self.ydim
                        xn[xn < 1] += self.xdim
                        xn[xn > self.xdim] -= self.xdim

                    elif self.maptype == 'standard':

                        # Take only valid nodes
                        idx = np.where((yn >= 1) & (yn <= self.ydim) & (xn >= 1) & (xn <= self.xdim))
                        xn = xn[idx]
                        yn = yn[idx]

                    else:
                        raise ValueError('No such map type!')

                    # Set neighborhood
                    NB[ind - 1, (yn - 1) * self.xdim + xn - 1] = 1


            NBnonzero = np.array([np.nonzero(NB[i, :])[0] for i in range(self.xdim * self.ydim)])

        self.NB = NB
        self.NBNZ = NBnonzero.astype(np.uint8)

        if verbose:
            print(f'Generated neighborhood of shape [{self.xdim}, {self.ydim}], maptype {self.maptype} \n'
                  f'Neighborhood shape is a {self.nbshape} of size {self.nbsize}')



def get_matlab_data(samples = 50000, dataNum = 13,
                    directory:str = '/home/locadmin/Documents/imageica/data/',
                    winsize = (16, 16), verbose = True):
    """
    Adapting the random sampling from the MATLAB package

    in order to sample from random images,
    the images should be .tiff files numbered 1 - dataNum in directory

    :param samples:
        number of samples to take
    :param dataNum:
        number of images to sample from
    :param directory:
        string containing the path to the directory containing images to sample from
    :param winsize:
        tuple of [xdim, ydim] containing the size of the sample window
    """
    X = np.zeros(shape = (winsize[0]*winsize[1], samples))
    sampleNum = 0
    for i in range(dataNum):
        if verbose:
            print(f'Sampling image {i+1}...')
        # Even things out (take enough from last image)
        if i == dataNum-1:
            getsample = int(samples/dataNum) + samples-(int(samples/dataNum)*dataNum)
        else: getsample = int(samples/dataNum)

        I = cv2.imread(f'{directory}{i+1}.tiff', 0)
        I = I.astype(np.float64)
        # normalize to zero mean and unit variance
        I -= np.mean(np.mean(I))
        I = I/np.sqrt(np.mean(np.mean(np.square(I))))

        posy = np.random.randint(0, I.shape[0]-winsize[0], getsample)
        posx = np.random.randint(0, I.shape[1]-winsize[1], getsample)
        for j in range(getsample):
            X[:, sampleNum] = (I[posy[j]:posy[j]+winsize[0], posx[j]:posx[j]+winsize[1]]).flatten()
            sampleNum +=1

    return X