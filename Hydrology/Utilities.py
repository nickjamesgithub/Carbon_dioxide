import numpy as np
import scipy as sp
from scipy import stats
from scipy.fftpack import fft, fftshift, ifft, ifftshift
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error

def sqdist_gramix(x_p, x_q, length_scale):
    """
    Compute a gram matrix of euclidean distances between two datasets under an isotropic or anisotropic length scale.

    Parameters
    ----------
    x_p : np.ndarray [Size: (n_p, d)]
        A dataset
    x_q : np.ndarray [Size: (n_q, d)]
        A dataset
    length_scale : float or np.ndarray [Size: () or (1,) for isotropic; (d,) for anistropic]
        The length scale(s)

    Returns
    -------
    np.ndarray [Size: (n_p, n_q)]
        The resulting gram matrix
    """
    # Size: (n_p, d)
    z_p = np.atleast_2d(x_p) / length_scale
    # Size: (n_q, d)
    z_q = np.atleast_2d(x_q) / length_scale
    # Size: (n_p, n_q)
    d_pq = np.dot(z_p, z_q.transpose())
    # Size: (n_p,)
    d_p = np.sum(z_p ** 2, axis=1)
    # Size: (n_q,)
    d_q = np.sum(z_q ** 2, axis=1)
    # Size: (n_p, n_q)
    return d_p[:, np.newaxis] - 2 * d_pq + d_q

def spline(x_p, x_q, var):

    # Size: (n_p, d)
    z_p = np.atleast_2d(x_p)
    # Size: (n_q, d)
    z_q = np.atleast_2d(x_q)
    with tf.name_scope('spline'):
        K = np.minimum(z_p, z_p.transpose())
        # K[:,0] = 1
    K = np.array(K)
    return (1/var) * K

def linear_kernel(x_p, x_q, theta, name=None):
    """
    Define the linear kernel.
    The linear kernel does not need any hyperparameters.
    Passing hyperparameter arguments do not change the kernel behaviour.
    Parameters
    ----------
    x_p : tensorflow.Tensor
        A dataset of size (n_p, d)
    x_q : tensorflow.Tensor
        A dataset of size (n_p, d)
    Returns
    -------
    tensorflow.Tensor
        The gram matrix (n_p, n_q)
    """
    with tf.name_scope('linear_kernel'):
        return tf.multiply(tf.square(theta), tf.matmul(x_p, tf.transpose(x_q)), name=name)


def sqdist(x_p, x_q, theta):
    """
    Pairwise squared Euclidean distanced between datasets under length scaling.
    Parameters
    ----------
    x_p : tensorflow.Tensor
        A dataset of size (n_p, d)
    x_q : tensorflow.Tensor
        A dataset of size (n_p, d)
    theta : tensorflow.Tensor
        Length scale(s) for scaling the distance [(), (1,), (d,)]
    Returns
    -------
    tensorflow.Tensor
        The pairwise squared Euclidean distance under length scaling (n_p, n_q)
    """
    with tf.name_scope('sqdist'):
        z_p = tf.divide(x_p, theta)  # (n_p, d)
        z_q = tf.divide(x_q, theta)  # (n_q, d)
        d_pq = tf.matmul(z_p, tf.transpose(z_q))  # (n_p, n_q)
        d_p = tf.reduce_sum(tf.square(z_p), axis=1)  # (n_p,)
        d_q = tf.reduce_sum(tf.square(z_q), axis=1)  # (n_q,)
        return d_p[:, tf.newaxis] - 2 * d_pq + d_q  # (n_p, n_q)

def gaussian_kernel_gramix(x_p, x_q, length_scale, var):
    """
    Compute a gram matrix of Gaussian kernel values between two datasets under an isotropic or anisotropic length scale.

    Parameters
    ----------
    x_p : np.ndarray [Size: (n_p, d)]
        A dataset
    x_q : np.ndarray [Size: (n_q, d)]
        A dataset
    length_scale : float or np.ndarray [Size: () or (1,) for isotropic; (d,) for anistropic]
        The length scale(s)

    Returns
    -------
    np.ndarray [Size: (n_p, n_q)]
        The resulting gram matrix
    """
    # Size: (n_p, n_q)
    return var * np.exp(-0.5 * sqdist_gramix(x_p, x_q, length_scale))

# Generate covariance Matrix
def squared_exponential_covariance(x, y, l_scale, var):
    omega = np.zeros((len(x),len(y)))
    for i in range(len(x)):
        for j in range(len(y)):
            omega[i, j] = var * np.exp(-1 / (2 * l_scale ** 2) * ((x[i] - y[j]) ** 2))
    return omega

def posterior_predictive_sqe(y, train, test, log_l_scale, log_var, log_noise_var):
    l_scale = np.exp(log_l_scale)
    var = np.exp(log_var)
    noise_var = np.exp(log_noise_var)

    k_xx = gaussian_kernel_gramix(train, train, l_scale, var) + np.eye(len(train)) * noise_var
    k_x_xstar = gaussian_kernel_gramix(train, test, l_scale, var)
    k_xstar_x = gaussian_kernel_gramix(test, train, l_scale, var)
    k_xstar_xstar = gaussian_kernel_gramix(test, test, l_scale, var)

    # Generate Covariance matrix - predictive distribution
    f_star = np.dot(k_xstar_x, np.dot(np.linalg.inv(k_xx), y))
    cov_f_star = k_xstar_xstar - np.dot(np.dot(k_xstar_x, np.linalg.inv(k_xx)), k_x_xstar)
    return f_star, cov_f_star


def posterior_predictive_spline(y, train, test, log_var, log_noise_var):
    var = np.exp(log_var)
    noise_var = np.exp(log_noise_var)

    k_xx = gaussian_kernel_gramix(train, train, var) + np.eye(len(train)) * noise_var
    k_x_xstar = gaussian_kernel_gramix(train, test, var)
    k_xstar_x = gaussian_kernel_gramix(test, train, var)
    k_xstar_xstar = gaussian_kernel_gramix(test, test, var)

    # Generate Covariance matrix - predictive distribution
    f_star = np.dot(k_xstar_x, np.dot(np.linalg.inv(k_xx), y))
    cov_f_star = k_xstar_xstar - np.dot(np.dot(k_xstar_x, np.linalg.inv(k_xx)), k_x_xstar)
    return f_star, cov_f_star

def log_likelihood_sqe(y, f_star, cov_f_star):
    n = len(y)
    mean = f_star
    cov = cov_f_star #+ np.identity(n)*.1
    return -(n/2) * np.log(np.linalg.det(cov)) - 0.5 * np.sum(np.matmul(np.transpose(y-mean), np.matmul(np.linalg.inv(cov), (y-mean))))

def nu_p(hyperparameter):
    noise = sp.stats.norm.rvs(0, .3)
    rvs = hyperparameter + noise
    return rvs

def nu_noise_var(hyperparameter):
    noise = sp.stats.norm.rvs(0, .3)
    rvs = hyperparameter + noise
    return rvs

def logpdf_lscale(log_l_scale): # Note scale is exp(mu) and s = sigma
    # logpdf = sp.stats.lognorm.logpdf(log_l_scale, s = 2, scale=np.exp(10))
    logpdf = sp.stats.norm.logpdf((log_l_scale), loc=1, scale=.05) # 1,15  10, 2 for AR1 and AR2 and this is exponent
    return logpdf

def logpdf_var(log_var):
    # logpdf = sp.stats.lognorm.logpdf(log_var, s = 3, scale=3)
    logpdf = sp.stats.norm.logpdf((log_var), loc=1, scale=.1)
    return logpdf

def logpdf_alpha(alpha):
    # pdf = sp.stats.lognorm.logpdf(alpha, s=5, scale=1)
    pdf = sp.stats.norm.logpdf((alpha), loc=1, scale=.1) # 1,2
    return pdf

def logpdf_C(C):
    pdf = sp.stats.norm.logpdf((C), loc=1, scale=.1) # 1,1
    return pdf

def logpdf_noisevar(log_var):
    noise_var = np.exp(log_var)
    # logpdf = sp.stats.lognorm.logpdf(noise_var, s = 3, scale=2)
    logpdf = sp.stats.norm.logpdf((noise_var), loc=10, scale=.1) # 3,2
    return logpdf

def logpdf_d(d):
    logpdf = sp.stats.uniform.pdf(d, 1, 10)
    return logpdf

def proposal_distribution(input):
    return input

def draw_sigma_squared(a,b):
    return sp.stats.invgamma.rvs(a)*b

def model_neural_network_kernel(x_p, x_q, alpha, C, var):
    """1D version with the signal variance"""
    return var * neural_network_kernel_vectorized(x_p, x_q, alpha, C)

def neural_network_kernel_vectorized(x_p, x_q, alpha, c): # This is the hyperbolic tangent Kernel
    """
    :param x_p:
        Dataset of (n_p, d)
    :param x_q:
        Dataset of (n_q, d)
    :param sigma: # This is one hyperparameter for a 1d problem. That is a 1x1 matrix
        Covariance Matrix of (d, d)
    :return:
        Kernel Matrix of (n_p, n_q)
    """

    K = np.arctan(1/alpha * np.dot(x_p, np.transpose(x_q)) + c)
    return K

def polynomial_kernel_vectorized(x_p, x_q, alpha, c, d, var):
    K = var * (1/alpha * np.dot(x_p, np.transpose(x_q)) + c)**d
    return K

# Define Likelihood Function
def whittlelikelihood_sqe(y, train, test, log_l_scale, noise_var): # y should be the log adjusted periodogram
    f_star, cov_f_star = posterior_predictive_sqe(y, train, test, log_l_scale, noise_var)
    g = f_star
    likelihood = -np.sum(g + np.exp(y)/np.exp(g))
    return likelihood

def generate_periodogram(time_series):
    periodogram = np.log(np.abs(fft(fftshift(time_series)))**2)
    return periodogram

def gen_ar1(ar_inputs):
    size = len(ar_inputs[0])
    y_0 = ar_inputs[1]
    sigma = ar_inputs[2]
    alpha1 = ar_inputs[3]
    e = sigma*np.random.randn(size)
    x = np.zeros(size)
    for i in np.arange(1,size,1):
        if i == 1:
            x[i] = y_0
        else:
            x[i] = alpha1*x[i-1] + e[i]
    return x

def gen_ar_custom(ar_inputs):
    size = len(ar_inputs[0])
    y_0 = ar_inputs[1]
    sigma = ar_inputs[2]
    alpha1 = ar_inputs[3]
    alpha2 = ar_inputs[4]
    e = sigma*np.random.randn(size)
    x = np.zeros(size)
    for i in np.arange(1,size,1):
        if i == 1:
            x[i] = y_0
        if i == 2:
            x[i] = y_0
        else:
            x[i] = - alpha1*x[i-12] - alpha2*x[i-1] - alpha1*alpha2*x[i-30] + e[i]
    return x

def gen_ar2(ar_inputs):
    size = len(ar_inputs[0])
    y_0 = ar_inputs[1]
    sigma = ar_inputs[2]
    alpha1 = ar_inputs[3]
    alpha2 = ar_inputs[4]
    e = sigma*np.random.randn(size)
    x = np.zeros(size)
    for i in np.arange(1,size,1):
        if i == 1:
            x[i] = y_0
        if i == 2:
            x[i] = y_0
        else:
            x[i] = alpha1*x[i-1] + alpha2*x[i-2] + e[i]
    return x

def gen_ar4(ar_inputs):
    size = len(ar_inputs[0])
    y_0 = ar_inputs[1]
    sigma = ar_inputs[2]
    alpha1 = ar_inputs[3]
    alpha2 = ar_inputs[4]
    alpha3 = ar_inputs[5]
    alpha4 = ar_inputs[6]
    e = sigma * np.random.randn(size)
    x = np.zeros(size)
    for i in np.arange(1, size, 1):
        if i == 1:
            x[i] = y_0
        if i == 2:
            x[i] = y_0
        if i == 3:
            x[i] = y_0
        if i == 4:
            x[i] = y_0
        else:
            x[i] = alpha1 * x[i - 1] + alpha2 * x[i - 2] + alpha3 * x[i - 3] + alpha4 * x[i - 4] + e[i]
    return x

# Define AR(2) Process
def ts_gen_ar2(size, sigma, alpha2):
    e = sigma*np.random.randn(size)
    x = np.zeros(size)
    for i in np.arange(1,size,1):
        for j in np.linspace(0,size,size):
            x[i] = 0.8*(1-0.5*np.cos(np.pi*j/1024))*x[i-1] + alpha2*x[i-2] + e[i]
    return x

# Define AR(3) Model
def ts_gen_ar3(size, sigma, alpha1, alpha2, alpha3):
    e = sigma*np.random.randn(size)
    x = np.zeros(size)
    for i in np.arange(1,size,1):
        x[i] = alpha1*x[i-1] + alpha2*x[i-2] +alpha3*x[i-1]+ e[i]
    return x

# Define AR(2) Function
def ts_gen_ar2(size, sigma, alpha1, alpha2):
    e = sigma*np.random.randn(size)
    x = np.zeros(size)
    for i in np.arange(1,size,1):
        x[i] = alpha1*x[i-1] + alpha2*x[i-2] + e[i]
    return x

def derivative_var(x_p, x_q, var_f, lengthscale):
    return 2 * np.sqrt(var_f) * np.exp(-0.5 * sqdist_gramix(x_p, x_q, lengthscale))

def derivative_lscale(x_p, x_q, var_f, lengthscale):
    return var_f * sqdist_gramix(x_p, x_q, lengthscale)

def matern32(x_p, x_q, theta, name=None):
    """
    Define the Matern 3/2 kernel.
    The hyperparameters are the length scales of the kernel.
    Parameters
    ----------
    x_p : tensorflow.Tensor
        A dataset of size (n_p, d)
    x_q : tensorflow.Tensor
        A dataset of size (n_p, d)
    theta : tensorflow.Tensor
        Length scale(s) for scaling the distance [(), (1,), (d,)]
    Returns
    -------
    tensorflow.Tensor
        The gram matrix (n_p, n_q)
    """
    with tf.name_scope('matern32_kernel'):
        r = tf.sqrt(sqdist(x_p, x_q, theta))
        return tf.multiply(1 + r, tf.exp(-r), name=name)

def matern32_vectorized(x_p, x_q, l_scale, variance, name=None):
    with tf.name_scope('matern32_kernel'):
        r = np.sqrt(sqdist_gramix(x_p, x_q, l_scale))
        return variance * (1 + np.sqrt(3) * r) * np.exp(-np.sqrt(3) * r)

def matern52_vectorized(x_p, x_q, l_scale, variance, name=None):
    with tf.name_scope('matern52_kernel'):
        r = np.sqrt(sqdist_gramix(x_p, x_q, l_scale))
        r_2 = sqdist_gramix(x_p, x_q, l_scale)
        return variance * (1 + np.sqrt(5)*r + ((5/3)*r_2)) * np.exp(-np.sqrt(5)*r)

def s_matern32(x_p, x_q, theta, name=None):
    """
    Define the Matern 3/2 kernel.
    The hyperparameters are the sensitivity and length scales of the kernel.
    Parameters
    ----------
    x_p : tensorflow.Tensor
        A dataset of size (n_p, d)
    x_q : tensorflow.Tensor
        A dataset of size (n_p, d)
    theta : tensorflow.Tensor
        Length scale(s) for scaling the distance [(), (1 + 1,), (1 + d,)]
    Returns
    -------
    tensorflow.Tensor
        The gram matrix (n_p, n_q)
    """
    with tf.name_scope('s_matern32_kernel'):
        s = theta[0]
        l = theta[1:]
        return tf.multiply(tf.square(s), matern32(x_p, x_q, l), name=name)

def analytic_spectrum_ar1(sigma, phi_1, frequencies):
    numerator = (sigma**2)
    denominator = 1 + (phi_1**2) - (2*phi_1*np.cos(2*np.pi*frequencies))
    return numerator/denominator

def analytic_spectrum_ar2(sigma, phi_1, phi_2, frequencies):
    numerator = sigma**2
    denominator = (1 + phi_1**2 + phi_2**2 - 2*phi_1*(1-phi_2)*np.cos(2*np.pi*frequencies) - 2*phi_2*np.cos(4*np.pi*frequencies))
    return numerator/denominator

def eigen_decomposition(omega, num_eigen):
    vals, vecs = np.linalg.eig(omega)
    diag = np.diag(np.sqrt(vals))
    X_complete = np.dot(vecs, diag)  # Full X matrix
    vals = vals[0:num_eigen]
    vecs = vecs[:, 0:num_eigen]
    diag = diag[0:num_eigen, 0:num_eigen]
    X_eigen = np.dot(vecs, diag)
    return np.nan_to_num(X_eigen), np.nan_to_num(diag)

def rmse(y, y_hat):
    return np.sqrt(mean_squared_error(y, y_hat))

def log_likelihood_exponential(mcmc_values, ground_truth, maximum_likelihood):
    n = len(mcmc_values)
    logpdf = np.log(maximum_likelihood) - maximum_likelihood*ground_truth
    return logpdf

def skl(ground_truth, spectral_estimate):
    kl = np.sum(ground_truth*np.log(ground_truth/spectral_estimate)) + np.sum(spectral_estimate*np.log(spectral_estimate/ground_truth))
    return kl

def MAE(gt, estimate):
    return mean_absolute_error(gt, estimate)

def custom_spectrum(sigma_squared, f):
    numerator = sigma_squared
    # denominator = np.abs(1 + .2*np.exp(-2j*np.pi*12*f) + 0.5*np.exp(-2j*np.pi*f) + 0.1*np.exp(-2j*np.pi*f*13))**2
    denominator = np.abs(1 + .5 * np.exp(-2j * np.pi * 12 * f) -.2 * np.exp(-2j * np.pi * f) - .1 * np.exp(-2j * np.pi * f * 30)) ** 2
    return numerator/denominator

def _error(actual: np.ndarray, predicted: np.ndarray):
    """ Simple error """
    return actual - predicted

def inrse(actual: np.ndarray, predicted: np.ndarray):
    """ Integral Normalized Root Squared Error """
    return np.sqrt(np.sum(np.square(_error(actual, predicted))) / np.sum(np.square(actual - np.mean(actual))))

def whittle_likelihood_spline(y, beta, X_prime, alpha_0, sigma_squared, tau_squared):
    n = len(y)
    XB = np.dot(X_prime, beta)
    mean = XB + alpha_0
    likelihood = -0.5*np.sum(alpha_0 + np.dot(X_prime,beta) + np.exp(y-alpha_0-np.dot(X_prime,beta)))- (alpha_0**2)/(2*sigma_squared) - 1/(2*tau_squared) * np.dot(np.transpose(beta), beta)
    return likelihood

def alpha_pdf(delta_y, alpha_0, sigma_squared):
    return 1/np.sqrt(2*np.pi*sigma_squared) * np.exp(-0.5*(np.sum(delta_y-alpha_0)**2)/sigma_squared)

def draw_sigma_squared(a,b):
    return sp.stats.invgamma.rvs(a, scale=1/b)

def beta_logpdf(beta_p, tau_squared_p):
    return np.exp(-.5 * np.matmul(np.transpose(beta_p), beta_p)/tau_squared_p)

def tau_squared_draw(tau_squared_c):
    nu_p = tau_squared_c + sp.stats.norm.rvs(0,.5)
    return nu_p

def tau_squared_logpdf(tau_squared, u, v):
    logpdf = sp.stats.invgamma.logpdf(tau_squared, u)*v
    return logpdf

def beta_logpdf(beta, tau_squared):
    logpdf = sp.stats.multivariate_normal.logpdf(beta, mean=np.zeros(len(beta)), cov=tau_squared*np.identity(len(beta)))
    return logpdf

def beta_proposal(beta):
    logpdf = sp.stats.multivariate_normal.logpdf(beta, mean=np.zeros(len(beta)), cov=0.1*np.identity(len(beta)))
    return logpdf

def tau_squared_logpdf(tau_squared,a,b):
    return sp.stats.invgamma.logpdf(tau_squared,a)*b

def IAE(gt_spectrum, est_spectrum):
    return np.sum(np.abs(gt_spectrum-est_spectrum))

def argmax_frequency_distance(y_hat, y, freqs):
    y_hat_freqs = freqs[np.argmax(y_hat)]
    y_freqs = freqs[np.argmax(y)]
    distance = np.abs(y_hat_freqs-y_freqs)
    return distance

def max_amplitude_distance(y_hat, y):
    y_hat_max = np.max(y_hat)
    y_max = np.max(y)
    distance = np.abs(y_hat_max-y_max)
    return distance

def complex_power_spectrum(sigma_squared, phi1, phi2, phi3, phi4, freqs):
    num = sigma_squared
    denom = np.abs(1-((phi1*np.exp(2*np.pi*freqs*1j*1)) + (phi2*np.exp(2*np.pi*freqs*1j*2)) + (phi3*np.exp(2*np.pi*freqs*1j*3)) + (phi4*np.exp(2*np.pi*freqs*1j*4))))**2
    return num/denom