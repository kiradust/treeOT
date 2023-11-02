import numpy as np
import scipy.optimize as spopt
import scipy.spatial.distance as spsd

def sinkhorn_single_update(v, a, b, K, log_errs=False):
    """
    Partly from NeuroHackAcademy: https://github.com/alecgt/otml-neurohackademy-2019/blob/master/lab/notebooks/lab_main.ipynb
    """
    row_err = col_err = None

    u = a / np.dot(K,v)
    if log_errs:
        col_err = np.linalg.norm((v * np.dot(K.T, u) - b),1)

    v = b / np.dot(K.T, u)
    if log_errs:
        row_err = np.linalg.norm((u * np.dot(K, v) - a), 1)
    return u, v, row_err, col_err

def sinkhorn_basic(a, b, K, max_iter = 1000, log_errs = True):
    """
    Partly from NeuroHackAcademy: https://github.com/alecgt/otml-neurohackademy-2019/blob/master/lab/notebooks/lab_main.ipynb
    """
    v = np.ones(len(b))
    errs = [(1,1)]
    i = 0
    logsum_err = 0

    while i < max_iter and logsum_err>-33:
        u, v, row_err, col_err = sinkhorn_single_update(v, a, b, K, log_errs)
        errs += [(row_err, col_err)]
        logsum_err = np.log(np.sum(errs[-20:-1]))
        i += 1
    return u, v, errs

def ma_inverseOT(P, mu, nu, niter = 1000, eps = 0.01, log_err = True):
    """
    P is mxn; mu m dim; alpha m dim; vu n dim; beta n dim
    Largely from Ma et al., 2020 (arXiv:2002.09650v2) -- discrete case
    implementation (tangentially) from https://github.com/El-Zag/Inverse-Optimal-Transport/blob/main/2.4%20Learning%20Cost%20Function.ipynb
    """
    m = mu.size
    n = nu.size
    cost = np.random.random((m, n)) #initialisation -- random weights
    alpha = mu
    beta = nu
    u = np.exp(alpha/eps)
    v = np.exp(beta/eps)
    errs = [(1,1)]
    i = 0
    logsum_err = 0

    while i < niter and logsum_err>-42:
        K=np.exp(-cost/eps)
        u, v, row_err, col_err = sinkhorn_single_update(v, mu, nu, K, log_errs = log_err)
        K=P/(np.outer(u,v.T))
        cost=-eps*np.log(K)
        errs += [(row_err, col_err)]
        logsum_err = np.log(np.sum(errs[-20:-1]))
        i += 1
        # optional regularization
        if n == m:
            cost = (cost + cost.T)/2
            #np.fill_diagonal(C, 0)
    return cost, errs

def demo_wasserstein(x, p, q):
    """
    Computes order-2 Wasserstein distance between two
    discrete distributions. 
    
    From Alex Williams: http://alexhwilliams.info/itsneuronalblog/2020/10/09/optimal-transport/

    Parameters
    ----------
    x : ndarray, has shape (num_bins, dimension)
    
        Locations of discrete atoms (or "spatial bins")

    p : ndarray, has shape (num_bins,)

        Probability mass of the first distribution on each atom.

    q : ndarray, has shape (num_bins,)

        Probability mass of the second distribution on each atom.

    Returns
    -------
    dist : float

        The Wasserstein distance between the two distributions.

    T : ndarray, has shape (num_bins, num_bins)

        Optimal transport plan. Satisfies p == T.sum(axis=0)
        and q == T.sum(axis=1).

    Note
    ----
    This function is meant for demo purposes only and is not
    optimized for speed. It should still work reasonably well
    for moderately sized problems.
    """

    # Check inputs.
    if (abs(p.sum() - 1) > 1e-9) or (abs(p.sum() - q.sum()) > 1e-9):
        raise ValueError("Expected normalized probability masses.")

    if np.any(p < 0) or np.any(q < 0):
        raise ValueError("Expected nonnegative mass vectors.")

    if (x.shape[0] != p.size) or (p.size != q.size):
        raise ValueError("Dimension mismatch.")

    # Compute pairwise costs between all xs.
    n, d = x.shape
    C = spsd.squareform(spsd.pdist(x, metric="sqeuclidean"))

    # Scipy's linear programming solver will accept the problem in
    # the following form:
    # 
    # minimize     c @ t        over t
    # subject to   A @ t == b
    #
    # where we specify the vectors c, b and the matrix A as parameters.

    # Construct matrices Ap and Aq encoding marginal constraints.
    # We want (Ap @ t == p) and (Aq @ t == q).
    Ap, Aq = [], []
    z = np.zeros((n, n))
    z[:, 0] = 1

    for i in range(n):
        Ap.append(z.ravel())
        Aq.append(z.transpose().ravel())
        z = np.roll(z, 1, axis=1)

    # We can leave off the final constraint, as it is redundant.
    # See Remark 3.1 in Peyre & Cuturi (2019).
    A = np.row_stack((Ap, Aq))[:-1]
    b = np.concatenate((p, q))[:-1]

    # Solve linear program, recover optimal vector t.
    result = spopt.linprog(C.ravel(), A_eq=A, b_eq=b)

    # Reshape optimal vector into (n x n) transport plan matrix T.
    T = result.x.reshape((n, n))

    # Return Wasserstein distance and transport plan.
    return np.sqrt(np.sum(T * C)), T