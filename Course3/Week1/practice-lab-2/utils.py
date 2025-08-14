import numpy as np
import matplotlib.pyplot as plt

def load_data():
    X = np.load("data/X_part1.npy")
    X_val = np.load("data/X_val_part1.npy")
    y_val = np.load("data/y_val_part1.npy")
    return X, X_val, y_val

def load_data_multi():
    X = np.load("data/X_part2.npy")
    X_val = np.load("data/X_val_part2.npy")
    y_val = np.load("data/y_val_part2.npy")
    return X, X_val, y_val


def multivariate_gaussian(X, mu, var):
    """
    Computes the probability 
    density function of the examples X under the multivariate gaussian 
    distribution with parameters mu and var. If var is a matrix, it is
    treated as the covariance matrix. If var is a vector, it is treated
    as the var values of the variances in each dimension (a diagonal
    covariance matrix
    """
    
    k = len(mu)
    
    if var.ndim == 1:
        var = np.diag(var)
        
    X = X - mu
    p = (2* np.pi)**(-k/2) * np.linalg.det(var)**(-0.5) * \
        np.exp(-0.5 * np.sum(np.matmul(X, np.linalg.pinv(var)) * X, axis=1))
    
    return p
        
def visualize_fit(X, mu, var):
    """
    This visualization shows you the 
    probability density function of the Gaussian distribution. Each example
    has a location (x1, x2) that depends on its feature values.
    """

    '''
    You have a dataset X with two features (e.g., Latency and Throughput).
    You already estimated its Gaussian parameters: mean (mu) and variance (var).
    This function plots:
        1) The data points.
        2) Contour lines showing regions of equal probability density (like a topographic map for probability).
    '''
    
    X1, X2 = np.meshgrid(np.arange(0, 35.5, 0.5), np.arange(0, 35.5, 0.5))
    # np.meshgrid makes a 2D grid covering the range 0 to 35 (in steps of 0.5) for both axes.
    # Think of this as creating a map of coordinates where you want to evaluate your Gaussian function.
    # X1 and X2 are 2D arrays representing all possible (x₁, x₂) positions.
    Z = multivariate_gaussian(np.stack([X1.ravel(), X2.ravel()], axis=1), mu, var)
    '''
    Step 1 — What we have
    From the earlier step:
        X1, X2 = np.meshgrid(np.arange(0, 35.5, 0.5),
                            np.arange(0, 35.5, 0.5))
    X1 and X2 are 2D arrays of the same shape.
    Example with a tiny mesh:
        X1 =
        [[0, 1, 2],
        [0, 1, 2]]

        X2 =
        [[10, 10, 10],
        [11, 11, 11]]
    Each (X1[i,j], X2[i,j]) is one coordinate.

    Step 2 — Flattening
        X1.ravel()
    Turns X1 into a 1D array: [0, 1, 2, 0, 1, 2]
        X2.ravel()
    Also a 1D array: 10, 10, 10, 11, 11, 11]
    Step 3 — Stacking
        np.stack([X1.ravel(), X2.ravel()], axis=1)
    np.stack([...], axis=1) puts them side by side as columns.

    Result:
        [[ 0, 10],
        [ 1, 10],
        [ 2, 10],
        [ 0, 11],
        [ 1, 11],
        [ 2, 11]]
    ✅ Meaning:
    We now have an (N, 2) array where each row is a coordinate (x₁, x₂) in the grid.
    This is exactly the format we need to pass into multivariate_gaussian so it can compute p(x) for every point in the grid.
    '''
    Z = Z.reshape(X1.shape) #X1.shape: (71, 71)

    plt.plot(X[:, 0], X[:, 1], 'bx') #'bx' → blue x markers for each data point.

    if np.sum(np.isinf(Z)) == 0:
        plt.contour(X1, X2, Z, levels=10**(np.arange(-20., 1, 3)), linewidths=1)
    '''
    a) np.isinf(Z): Returns a boolean array of the same shape as Z.
    Each entry is:
        True if that value in Z is infinite (inf or -inf)
        False otherwise.
    Example:
        Z = [0.1, np.inf, 0.3]
        np.isinf(Z) → [False, True, False]
    b) np.sum(np.isinf(Z)) == 0
    np.sum() here counts how many True values there are (since True is treated as 1, False as 0).
    If the sum is 0, it means there are no infinite values in Z.
    This acts as a safety check to avoid plotting invalid data.

    c) plt.contour(...)
    If the data is valid:
        plt.contour(X1, X2, Z, levels=10**(np.arange(-20., 1, 3)), linewidths=1)
    X1, X2: meshgrid coordinates (the x–y positions on the plot).
    Z: probability values for each coordinate.
    levels=10**(np.arange(-20., 1, 3)):
    np.arange(-20., 1, 3) → [-20, -17, -14, …, -2, 1]
    10**(...) turns those into probability contour levels: [1e-20, 1e-17, ..., 1e-2, 1e1]

    This way, contours are drawn at logarithmically spaced probability values.

    linewidths=1: makes contour lines thin.

    ✅ Summary in plain English
    This code says:

    “If there are no infinite probability values in Z, plot contour lines showing where the Gaussian PDF has certain probability levels, spaced on a log scale from 
    10^(−20) up to 10^(1).”
    '''
    # Set the title
    plt.title("The Gaussian contours of the distribution fit to the dataset")
    # Set the y-axis label
    plt.ylabel('Throughput (mb/s)')
    # Set the x-axis label
    plt.xlabel('Latency (ms)')