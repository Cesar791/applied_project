import numpy as np
import math
import random


def predict(S, v, R, delta_t):
    """Propagate particles according to motion model. Add diffusion. """
    S_bar = np.zeros(S.shape)
    M = S.shape[1]
    nx = S.shape[0] - 1

    # Motion model.
    u = np.array([[0], [0]])

    randM = np.random.rand(nx, M)
    Rvert = np.diag(R)[np.newaxis, :].T # Diagonal elements of R stacked.

    diffusion = np.multiply(np.random.rand(nx, M), np.diag(R)[np.newaxis, :].T)

    S_bar[0:nx, :] = S[0:nx, :] + u + diffusion
    S_bar[nx, :] = S[nx, :]

    return S_bar


def systematic_resample(S_bar):
    """Perform systematic resampling of the particles. """
    M = S_bar.shape[1]
    nx = S_bar.shape[0] - 1
    S = np.zeros(S_bar.shape)

    density_function = np.cumsum(S_bar[nx, :])   # Cumulative density function.
    r0 = random.random()/M
    r = r0 + (np.arange(1, M + 1) - 1)/float(M) # r0 + (m - 1)/M

    A1 = np.tile(density_function, (M, 1))      # Repeat CDF vertically.
    A2 = np.tile(r[np.newaxis, :].T, (1, M))    # Repeat r horizontally.

    indices = np.argmax(A1 >= A2, axis = 1) # i = min CDF(j) >= r0 + (m - 1)/M

    S = S_bar[:, indices]   # Resample.

    S[nx, :] = 1/float(M)    # Reset weights.

    return S


def weight(S_bar, psi, outlier):
    """Weigh the particles according to the probabilities in psi. """
    psi_inliers = psi[np.invert(outlier), :]    # Discard outlier measurements.
    nx = S_bar.shape[0] - 1
    if psi_inliers.size > 0:

        weights = np.prod(psi_inliers, axis = 0)    # Multiply probabilities.
        weights = weights/np.sum(weights)           # Normalize weights.
        S_bar[nx, :] = weights

    return S_bar


def associate(S_bar, z, lambda_psi, Q):
    """Calculate probability of each particle given each measurement. """
    n = z.shape[1]
    M = S_bar.shape[1]
    nx = S_bar.shape[0] - 1
    dim = nx

    z_pred = np.tile(S_bar[0:dim, :], (1, n)) # [x x ... x]
    z_obs = np.reshape(np.repeat(z, M), (dim, n*M)) # [z1 ... z1 zn ... zn]

    nu = z_obs - z_pred # True observation minus predicted observation.

    exp_term = -0.5*np.sum(np.multiply(np.dot(nu.T, Q).T, nu), axis = 0)
    psis = 1/(2*math.pi*math.sqrt(np.linalg.det(Q)))*np.exp(exp_term)

    psi = np.reshape(psis, (n, M)) # Rows: measurements, columns: particles.

    outlier = np.mean(psi, axis = 1) < lambda_psi

    return outlier, psi


def main():

    S = np.array([
        [1, 2, 3, 4, 5],
        [1, 3, 2, 0, 1],
        [0.1, 0.2, 0.4, 0.25, 0.05]])

    v = 1.1
    R = np.diag([1.1, 1.2])
    delta_t = 0.1

    S_bar = predict(S, v, R, delta_t)

    Q = np.diag([0.4, 0.4])
    z = np.array([[1.5, 3], [1.4, 0]])
    lambda_psi = 0.001

    outlier, psi = associate(S_bar, z, lambda_psi, Q)

    S_bar = weight(S_bar, psi, outlier)

    S = systematic_resample(S_bar)




if __name__ == '__main__':
    main()
