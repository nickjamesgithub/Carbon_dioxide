import numpy as np
import scipy as sp
from scipy import stats
import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftshift, ifft, ifftshift
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
import pylab
from scipy.stats import wasserstein_distance
from scipy.sparse import csgraph
from scipy.sparse.linalg import eigsh
from scipy.spatial.distance import pdist, squareform
import numpy as np
from scipy.stats.kde import gaussian_kde
from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances
import decimal

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def levenshteinDistance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]

def distance_outputs(x,y):
    set_x = []
    set_y = []
    for i in range(len(x)):
        set_x.append(np.abs(x[i] - find_nearest(y, x[i])))
    for j in range(len(y)):
        set_y.append(np.abs(y[j] - find_nearest(x, y[j])))
    set_x = np.array(set_x)
    set_y = np.array(set_y)
    dist_vector = np.concatenate((set_x, set_y))
    hausdorff_distance = np.max(dist_vector)
    modified_hausdorff_distance = np.max((np.mean(set_x), np.mean(set_y)))  # see page 6 of distance between sets conci and kubrusly
    modified_hausdorff_distance_2 = (np.sum(set_x) + np.sum(set_y))  # Reference 16 & 17
    modified_hausdorff_distance_3 = (1 / (len(set_x) + len(set_y))) * ((np.sum(set_x) + np.sum(set_y)))  # Reference 16/17 geometric mean error
    mj05_distance = (((np.sum(set_x ** 0.5) / len(set_x)) + (np.sum(set_y ** 0.5) / len(set_y))) / 2) ** 2
    mj1_distance = ((np.mean(set_x) + np.mean(set_y)) / 2)  # Reference 17
    mj2_distance = np.sqrt(((np.sum(set_x ** 2) / len(set_x)) + (np.sum(set_y ** 2) / len(set_y))) / 2)
    wasserstein_dist = wasserstein_distance(x,y)

    return hausdorff_distance, modified_hausdorff_distance, modified_hausdorff_distance_2, \
           modified_hausdorff_distance_3, wasserstein_dist, mj05_distance, mj1_distance, mj2_distance


def distance_matrix_calculations(cps):
    grid_length = len(cps)
    # Standard Grid
    hausdorff_matrix = np.zeros((grid_length, grid_length))
    modified_hausdorff_2_matrix = np.zeros((grid_length, grid_length))
    modified_hausdorff_3_matrix = np.zeros((grid_length, grid_length))
    modified_hausdorff_matrix = np.zeros((grid_length, grid_length))
    wasserstein_matrix = np.zeros((grid_length, grid_length))
    mj05_matrix = np.zeros((grid_length, grid_length))
    mj1_matrix = np.zeros((grid_length, grid_length))
    mj2_matrix = np.zeros((grid_length, grid_length))

    # 1/(1+D) for each entry
    normalised_hausdorff_matrix = np.zeros((grid_length, grid_length))
    normalised_modified_hausdorff_matrix = np.zeros((grid_length, grid_length))
    normalised_modified_hausdorff_2_matrix = np.zeros((grid_length, grid_length))
    normalised_modified_hausdorff_3_matrix = np.zeros((grid_length, grid_length))
    normalised_wasserstein_matrix = np.zeros((grid_length, grid_length))
    normalised_mj05_matrix = np.zeros((grid_length, grid_length))
    normalised_mj1_matrix = np.zeros((grid_length, grid_length))
    normalised_mj2_matrix = np.zeros((grid_length, grid_length))

    for i in range(len(cps)):
        for j in range(len(cps)):
            hausdorff_matrix[i,j] = distance_outputs(cps[i], cps[j])[0] # 1/(1+d) normalise matrix
            modified_hausdorff_matrix[i,j] = distance_outputs(cps[i], cps[j])[1] # 1/(1+d) normalise matrix
            modified_hausdorff_2_matrix[i, j] = distance_outputs(cps[i], cps[j])[2]  # 1/(1+d) normalise matrix
            modified_hausdorff_3_matrix[i, j] = distance_outputs(cps[i], cps[j])[3]  # 1/(1+d) normalise matrix
            wasserstein_matrix[i,j] = distance_outputs(cps[i], cps[j])[4]
            mj05_matrix[i, j] = distance_outputs(cps[i], cps[j])[5]
            mj1_matrix[i, j] = distance_outputs(cps[i], cps[j])[6]
            mj2_matrix[i,j] = distance_outputs(cps[i], cps[j])[7]

    for i in range(len(cps)):
        for j in range(len(cps)):
            normalised_hausdorff_matrix[i,j] = np.nan_to_num(1 - (distance_outputs(cps[i], cps[j])[0])/np.max(hausdorff_matrix)) # Aij = 1 - Dij/max(D)
            normalised_modified_hausdorff_matrix[i, j] = np.nan_to_num(1 - (distance_outputs(cps[i], cps[j])[1]) / np.max(modified_hausdorff_matrix))
            normalised_modified_hausdorff_2_matrix[i, j] = np.nan_to_num(1 - (distance_outputs(cps[i], cps[j])[2]) / np.max(modified_hausdorff_2_matrix))
            normalised_modified_hausdorff_3_matrix[i, j] = np.nan_to_num(1 - (distance_outputs(cps[i], cps[j])[3]) / np.max(modified_hausdorff_3_matrix))
            normalised_wasserstein_matrix[i, j] = np.nan_to_num(distance_outputs(cps[i], cps[j])[4] / np.max(wasserstein_matrix))
            normalised_mj05_matrix[i, j] = np.nan_to_num(1 - (distance_outputs(cps[i], cps[j])[5]) / np.max(mj05_matrix))
            normalised_mj1_matrix[i, j] = np.nan_to_num(1 - (distance_outputs(cps[i], cps[j])[6])/np.max(mj1_matrix))
            normalised_mj2_matrix[i,j] = np.nan_to_num(1 - (distance_outputs(cps[i], cps[j])[7])/np.max(mj2_matrix))

    return hausdorff_matrix, modified_hausdorff_matrix,modified_hausdorff_2_matrix, modified_hausdorff_3_matrix, wasserstein_matrix, mj05_matrix, mj1_matrix, mj2_matrix, \
           normalised_hausdorff_matrix, normalised_modified_hausdorff_matrix, normalised_modified_hausdorff_2_matrix, normalised_modified_hausdorff_3_matrix, \
           normalised_wasserstein_matrix, normalised_mj05_matrix, normalised_mj1_matrix, normalised_mj2_matrix

def transitivity_test(grid, matrix, distance):
    triangle_distance = []
    omega = np.zeros((len(grid), len(grid), len(grid)))
    for i in range(len(grid)):
        for j in range(len(grid)):
            for k in range(len(grid)):
                if distance == "hausdorff":
                    ratio = matrix[i, k] / (matrix[i, j] + matrix[j, k])
                    omega[i, j, k] = np.nan_to_num(ratio)
                    triangle_distance.append(ratio)
                if distance == "modified_hausdorff":
                    ratio = matrix[i, k] / (matrix[i, j] + matrix[j, k])
                    omega[i, j, k] = np.nan_to_num(ratio)
                    triangle_distance.append(ratio)
                if distance == "modified_hausdorff_2":
                    ratio = matrix[i, k] / (matrix[i, j] + matrix[j, k])
                    omega[i, j, k] = np.nan_to_num(ratio)
                    triangle_distance.append(ratio)
                if distance == "modified_hausdorff_3":
                    ratio = matrix[i, k] / (matrix[i, j] + matrix[j, k])
                    omega[i, j, k] = np.nan_to_num(ratio)
                    triangle_distance.append(ratio)
                if distance == "mj05":
                    ratio = matrix[i, k] / (matrix[i, j] + matrix[j, k])
                    omega[i, j, k] = np.nan_to_num(ratio)
                    triangle_distance.append(ratio)
                if distance == "mj1":
                    ratio = matrix[i, k] / (matrix[i, j] + matrix[j, k])
                    omega[i, j, k] = np.nan_to_num(ratio)
                    triangle_distance.append(ratio)
                if distance == "mj2":
                    ratio = matrix[i, k] / (matrix[i, j] + matrix[j, k])
                    omega[i, j, k] = np.nan_to_num(ratio)
                    triangle_distance.append(ratio)
                if distance == "wasserstein":
                    ratio = matrix[i, k] / (matrix[i, j] + matrix[j, k])
                    omega[i, j, k] = np.nan_to_num(ratio)
                    triangle_distance.append(ratio)
    triangle_distance = np.nan_to_num(triangle_distance)
    distances = triangle_distance.flatten()
    fail_values_avg = np.nan_to_num(distances[np.where(distances > 1.01)])
    fail_percentage = np.nan_to_num(len(fail_values_avg)/len(omega.flatten()))
    return fail_percentage, fail_values_avg


def plot_3d(grid, distance_measure, data_generation):
    hausdorff_matrix, modified_hausdorff_matrix, modified_hausdorff_2_matrix, modified_hausdorff_3_matrix, wasserstein_matrix, mj05_matrix, mj1_matrix, mj2_matrix, \
    normalised_hausdorff_matrix, normalised_modified_hausdorff_matrix, normalised_modified_hausdorff_2_matrix, normalised_modified_hausdorff_3_matrix, \
    normalised_wasserstein_matrix, normalised_mj05_matrix, normalised_mj1_matrix, normalised_mj2_matrix = distance_matrix_calculations(grid)
    triangle_distance = []
    omega = np.zeros((len(grid), len(grid), len(grid)))
    for i in range(len(grid)):
        for j in range(len(grid)):
            for k in range(len(grid)):
                if distance_measure == "hausdorff":
                    ratio = hausdorff_matrix[i, k] / (hausdorff_matrix[i, j] + hausdorff_matrix[j, k])
                    omega[i, j, k] = np.nan_to_num(ratio)
                    triangle_distance.append(ratio)
                if distance_measure == "modified_hausdorff":
                    ratio = modified_hausdorff_matrix[i, k] / (modified_hausdorff_matrix[i, j] + modified_hausdorff_matrix[j, k])
                    omega[i, j, k] = np.nan_to_num(ratio)
                    triangle_distance.append(ratio)
                if distance_measure == "modified_hausdorff_2":
                    ratio = modified_hausdorff_2_matrix[i, k] / (modified_hausdorff_2_matrix[i, j] + modified_hausdorff_2_matrix[j, k])
                    omega[i, j, k] = np.nan_to_num(ratio)
                    triangle_distance.append(ratio)
                if distance_measure == "modified_hausdorff_3":
                    ratio = modified_hausdorff_3_matrix[i, k] / (modified_hausdorff_3_matrix[i, j] + modified_hausdorff_3_matrix[j, k])
                    omega[i, j, k] = np.nan_to_num(ratio)
                    triangle_distance.append(ratio)
                if distance_measure == "wasserstein":
                    ratio = wasserstein_matrix[i, k] / (wasserstein_matrix[i, j] + wasserstein_matrix[j, k])
                    omega[i, j, k] = np.nan_to_num(ratio)
                    triangle_distance.append(ratio)
                if distance_measure == "mj05":
                    ratio = mj05_matrix[i, k] / (mj05_matrix[i, j] + mj05_matrix[j, k])
                    omega[i, j, k] = np.nan_to_num(ratio)
                    triangle_distance.append(ratio)
                if distance_measure == "mj1":
                    ratio = mj1_matrix[i, k] / (mj1_matrix[i, j] + mj1_matrix[j, k])
                    omega[i, j, k] = np.nan_to_num(ratio)
                    triangle_distance.append(ratio)
                if distance_measure == "mj2":
                    ratio = mj2_matrix[i, k] / (mj2_matrix[i, j] + mj2_matrix[j, k])
                    omega[i, j, k] = np.nan_to_num(ratio)
                    triangle_distance.append(ratio)

    print(triangle_distance)
    if np.max(np.nan_to_num(triangle_distance)) > 1:
        print("Element fails triangle inequality")
    else:
        print("No elements fail triangle inequality")

    # Make this bigger to generate a dense grid.
    N = len(grid)

    # Create some random data.
    volume = np.random.rand(N, N, N)

    # Create the x, y, and z coordinate arrays.  We use
    # numpy's broadcasting to do all the hard work for us.
    # We could shorten this even more by using np.meshgrid.
    x = np.arange(omega.shape[0])[:, None, None]
    y = np.arange(omega.shape[1])[None, :, None]
    z = np.arange(omega.shape[2])[None, None, :]
    x, y, z = np.broadcast_arrays(x, y, z)

    # Set custom colour scheme
    c = np.tile(omega.ravel()[:, None], [1, 3])
    my_color = []
    for i in range(len(omega.ravel())):
        if omega.ravel()[i] <= 1.01:
            my_color.append('blue')
        if 1.01 < omega.ravel()[i] <= 2:
            my_color.append('yellow')
        if omega.ravel()[i] > 2:
            my_color.append('red')
    my_color = np.array(my_color)

    triangle_distance = np.nan_to_num(triangle_distance)
    distances = triangle_distance.flatten()
    fail_values_avg = np.nan_to_num(distances[np.where(distances > 1)])
    fail_percentage = np.nan_to_num(len(fail_values_avg)/len(omega.flatten()))

    print("fail values average", np.mean(fail_values_avg))
    print("fail percentage", fail_percentage)

    # my_color = np.where(omega.ravel() <= 1, 'blue', (np.where(1 < omega.ravel() < 2, 'yellow', 'red')))
    # col = np.where(x<1,'k',np.where(y<5,'b','r'))

    # col = np.where(x<1,'k',np.where(y<5,'b','r'))
    # Do the plotting in a single call.
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(x.ravel(),
               y.ravel(),
               z.ravel(),
               c=my_color)
    plt.savefig(distance_measure+data_generation+"Transitivity")
    plt.show()

def simulated_plots(ts1,ts2,ts3,ts4,ts5,ts6,ts7,ts8,ts9,ts10,combined_sets, data_generation):
    ax1 = plt.subplot(10,1,1)
    ax2 = plt.subplot(10,1,2)
    ax3 = plt.subplot(10,1,3)
    ax4 = plt.subplot(10,1,4)
    ax5 = plt.subplot(10,1,5)
    ax6 = plt.subplot(10,1,6)
    ax7 = plt.subplot(10,1,7)
    ax8 = plt.subplot(10,1,8)
    ax9 = plt.subplot(10,1,9)
    ax10 = plt.subplot(10,1,10)
    for i in range(len(ts1)):
        ax1.axvline(ts1[i], color='blue', alpha=1, linewidth=1)
        ax1.set_xlim([0, max(combined_sets)+1])
    for i in range(len(ts2)):
        ax2.axvline(ts2[i], color='red', alpha=1, linewidth=1)
        ax2.set_xlim([0, max(combined_sets) + 1])
    for i in range(len(ts3)):
        ax3.axvline(ts3[i], color='black', alpha=1, linewidth=1)
        ax3.set_xlim([0, max(combined_sets) + 1])
    for i in range(len(ts4)):
        ax4.axvline(ts4[i], color='orange', alpha=1, linewidth=1)
        ax4.set_xlim([0, max(combined_sets) + 1])
    for i in range(len(ts5)):
        ax5.axvline(ts5[i], color='blue', alpha=1, linewidth=1)
        ax5.set_xlim([0, max(combined_sets)+1])
    for i in range(len(ts6)):
        ax6.axvline(ts6[i], color='red', alpha=1, linewidth=1)
        ax6.set_xlim([0, max(combined_sets) + 1])
    for i in range(len(ts7)):
        ax7.axvline(ts7[i], color='black', alpha=1, linewidth=1)
        ax7.set_xlim([0, max(combined_sets) + 1])
    for i in range(len(ts8)):
        ax8.axvline(ts8[i], color='orange', alpha=1, linewidth=1)
        ax8.set_xlim([0, max(combined_sets) + 1])
    for i in range(len(ts9)):
        ax9.axvline(ts9[i], color='black', alpha=1, linewidth=1)
        ax9.set_xlim([0, max(combined_sets) + 1])
    for i in range(len(ts10)):
        ax10.axvline(ts10[i], color='orange', alpha=1, linewidth=1)
        ax10.set_xlim([0, max(combined_sets) + 1])
    #plt.savefig(data_generation+"ChangePoints")
    plt.show()

def measles_plots(ts1, ts2, ts3, ts4, ts5, ts6, ts7, combined_sets):
    ax1 = plt.subplot(7, 1, 1)
    ax2 = plt.subplot(7, 1, 2)
    ax3 = plt.subplot(7, 1, 3)
    ax4 = plt.subplot(7, 1, 4)
    ax5 = plt.subplot(7, 1, 5)
    ax6 = plt.subplot(7, 1, 6)
    ax7 = plt.subplot(7, 1, 7)

    for i in range(len(ts1)):
        ax1.axvline(ts1[i], color='blue', alpha=1, linewidth=1)
        ax1.set_xlim([0, max(combined_sets) + 1])
    for i in range(len(ts2)):
        ax2.axvline(ts2[i], color='red', alpha=1, linewidth=1)
        ax2.set_xlim([0, max(combined_sets) + 1])
    for i in range(len(ts3)):
        ax3.axvline(ts3[i], color='black', alpha=1, linewidth=1)
        ax3.set_xlim([0, max(combined_sets) + 1])
    for i in range(len(ts4)):
        ax4.axvline(ts4[i], color='orange', alpha=1, linewidth=1)
        ax4.set_xlim([0, max(combined_sets) + 1])
    for i in range(len(ts5)):
        ax5.axvline(ts5[i], color='blue', alpha=1, linewidth=1)
        ax5.set_xlim([0, max(combined_sets) + 1])
    for i in range(len(ts6)):
        ax6.axvline(ts6[i], color='red', alpha=1, linewidth=1)
        ax6.set_xlim([0, max(combined_sets) + 1])
    for i in range(len(ts7)):
        ax7.axvline(ts7[i], color='black', alpha=1, linewidth=1)
        ax7.set_xlim([0, max(combined_sets) + 1])
    plt.show()

def eigenspectrum(matrix, distance_measure, data_generation):
    # Eigen decomposition
    vals, vecs = np.linalg.eig(matrix)
    vecs = vecs[:, 0:10]
    eigenvalue_abs = (np.sort(np.abs(vals)))

    # Eigenvalue Absolute Value Plot
    plt.plot(eigenvalue_abs)
    # plt.title("Absolute Value of eigenvalues")
    plt.savefig(distance_measure+data_generation+"Eigenvalues")
    plt.show()

    return vals, vecs

def dendrogram_plot(matrix, distance_measure, data_generation, labels):

    # Compute and plot dendrogram.
    plt.rcParams.update({'font.size': 20})
    fig = pylab.figure(figsize=(15,10))
    axdendro = fig.add_axes([0.09,0.1,0.2,0.8]) # 0.1, 0.1, 0.2, 0.8
    Y = sch.linkage(matrix, method='centroid')
    # Z = sch.dendrogram(Y, orientation='right', labels=labels, leaf_rotation=360, leaf_font_size=7)
    Z = sch.dendrogram(Y, orientation='right', leaf_rotation=360, leaf_font_size=18, no_labels=True)
    axdendro.set_xticks([])
    # axdendro.set_yticks([])

    # Plot distance matrix.
    axmatrix = fig.add_axes([0.3,0.1,0.6,0.8])
    index = Z['leaves']
    D = matrix[index,:]
    D = D[:,index]
    im = axmatrix.matshow(D, aspect='auto', origin='lower')
    axmatrix.set_xticks([])
    axmatrix.set_yticks([])

    # Plot colorbar.
    axcolor = fig.add_axes([0.91,0.1,0.02,0.8])
    pylab.colorbar(im, cax=axcolor)
    plt.savefig(data_generation+distance_measure+"Dendrogram",bbox_inches="tight")
    plt.show()

    # Display and save figure.
    fig.show()

def rank_data(sorted_idx):
    # Ranks
    ranks = np.empty_like(sorted_idx)
    ranks[sorted_idx] = np.arange(len(sorted_idx))
    return ranks[sorted_idx]

def dendrogram_plot_labels(matrix, distance_measure, data_generation, labels):

    # Compute and plot dendrogram.
    plt.rcParams.update({'font.size': 8})
    fig = pylab.figure(figsize=(15,10))
    axdendro = fig.add_axes([0.11,0.1,0.2,0.8])
    Y = sch.linkage(matrix, method='centroid')
    Z = sch.dendrogram(Y, orientation='right', labels=labels, leaf_rotation=360, leaf_font_size=8, show_leaf_counts=False)
    # axdendro.set_xticks([])
    # axdendro.set_yticks([])

    # Plot distance matrix.
    axmatrix = fig.add_axes([0.3,0.1,0.6,0.8])
    index = Z['leaves']
    D = matrix[index,:]
    D = D[:,index]
    im = axmatrix.matshow(D, aspect='auto', origin='lower')
    axmatrix.set_xticks([])
    axmatrix.set_yticks([])

    # Plot colorbar.
    axcolor = fig.add_axes([0.91,0.1,0.02,0.8])
    pylab.colorbar(im, cax=axcolor)
    # plt.title(data_generation+distance_measure+"Dendrogram")
    plt.savefig(data_generation+distance_measure+"Dendrogram")
    plt.show()

    # Display and save figure.
    fig.show()


def spectral_clustering(normalised_matrix, clusters):
    # # Spectral Clustering
    # Adjacency Matrix in this case is weighted
    print("Weighted Adjacency Matrix")
    print(normalised_matrix)
    # diagonal matrix
    D = np.diag(normalised_matrix.sum(axis=1))

    # graph laplacian
    L = D-normalised_matrix

    # eigenvalues and eigenvectors
    vals, vecs = np.linalg.eig(L)

    # sort these based on the eigenvalues
    vecs = vecs[:,np.argsort(vals)]
    vals = vals[np.argsort(vals)]

    # kmeans on first three vectors with nonzero eigenvalues
    kmeans = KMeans(n_clusters=clusters)
    kmeans.fit((vecs[:,0:clusters]))
    colors = kmeans.labels_

    print("Clusters:", colors)

    return colors

def eigenDecomposition(affinity_matrix, plot=True):
    L = csgraph.laplacian(affinity_matrix, normed=True)
    n_components = affinity_matrix.shape[0]

    eigenvalues, eigenvectors = eigsh(L, k=n_components, which="LM", sigma=1.0, maxiter=5000)

    if plot:
        plt.title("largest eigenvalues of input matrix")
        plt.scatter(np.arange(len(eigenvalues)), eigenvalues)
        plt.grid()

    index_largest_gap = np.argmax(np.diff(eigenvalues))
    nb_clusters = index_largest_gap + 1
    return nb_clusters, eigenvalues, eigenvectors

def wasserstein_dist(x,y):
    return wasserstein_distance(x,y)

# This is effectively a collection of sets of change points
def synthetic_piecewise_mean_varying(means, variance, lengths):
    time_series = []
    for i in np.arange(0, len(lengths)):
        time_series.append(sp.norm.rvs(means[i], variance, lengths[i]).flatten())
    time_series = np.concatenate(time_series).ravel()
    return time_series

def synthetic_piecewise_variance_varying(means, variance, lengths):
    time_series = []
    for i in np.arange(0, len(lengths)):
        time_series.append(sp.norm.rvs(means, variance[i], lengths[i]).flatten())
    time_series = np.concatenate(time_series).ravel()
    return time_series

def generate_piecewise_mean(time_series_1, time_series_2):
    piece = []
    for i in range(len(time_series_2)):
        piece.append(time_series_1[i]*time_series_2[i])
    return piece

def compute_area(product_array, ts_length):
    product_area = np.sum(product_array)/ts_length
    return product_area

def Lp_norm(function, p):
    pth_power = np.abs(function)**p
    return (1/len(function) * np.sum(pth_power))**(1/p)


def predict_k(affinity_matrix):
    """
    Predict number of clusters based on the eigengap.
    Parameters
    ----------
    affinity_matrix : array-like or sparse matrix, shape: (n_samples, n_samples)
        adjacency matrix.
        Each element of this matrix contains a measure of similarity between two of the data points.
    Returns
    ----------
    k : integer
        estimated number of cluster.
    Note
    ---------
    If graph is not fully connected, zero component as single cluster.
    References
    ----------
    A Tutorial on Spectral Clustering, 2007
        Luxburg, Ulrike
        http://www.kyb.mpg.de/fileadmin/user_upload/files/publications/attachments/Luxburg07_tutorial_4488%5b0%5d.pdf
    """

    """
    If normed=True, L = D^(-1/2) * (D - A) * D^(-1/2) else L = D - A.
    normed=True is recommended.
    """
    normed_laplacian, dd = graph_laplacian(affinity_matrix, normed=True, return_diag=True)
    laplacian = _set_diag(normed_laplacian, 1, True)

    """
    n_components size is N - 1.
    Setting N - 1 may lead to slow execution time...
    """
    n_components = affinity_matrix.shape[0] - 1

    """
    shift-invert mode
    The shift-invert mode provides more than just a fast way to obtain a few small eigenvalues.
    http://docs.scipy.org/doc/scipy/reference/tutorial/arpack.html
    The normalized Laplacian has eigenvalues between 0 and 2.
    I - L has eigenvalues between -1 and 1.
    """
    eigenvalues, eigenvectors = eigsh(-laplacian, k=n_components, which="LM", sigma=1.0, maxiter=5000)
    eigenvalues = -eigenvalues[::-1]  # Reverse and sign inversion.

    max_gap = 0
    gap_pre_index = 0
    for i in range(1, eigenvalues.size):
        gap = eigenvalues[i] - eigenvalues[i - 1]
        if gap > max_gap:
            max_gap = gap
            gap_pre_index = i - 1

    k = gap_pre_index + 1

    return k

def rbf_kernel(X, Y=None, gamma=None):
    """
    Compute the rbf (gaussian) kernel between X and Y::
        K(x, y) = exp(-gamma ||x-y||^2)
    for each pair of rows x in X and y in Y.
    Read more in the :ref:`User Guide <rbf_kernel>`.
    Parameters
    ----------
    X : array of shape (n_samples_X, n_features)
    Y : array of shape (n_samples_Y, n_features)
    gamma : float, default None
        If None, defaults to 1.0 / n_features
    Returns
    -------
    kernel_matrix : array of shape (n_samples_X, n_samples_Y)
    """

    K = euclidean_distances(X, Y, squared=True)
    K *= -gamma
    np.exp(K, K)  # exponentiate K in-place
    return K

def mj1_distance(x,y):
    set_x = []
    set_y = []
    for i in range(len(x)):
        set_x.append(np.abs(x[i] - find_nearest(y, x[i])))
    for j in range(len(y)):
        set_y.append(np.abs(y[j] - find_nearest(x, y[j])))
    set_x = np.array(set_x)
    set_y = np.array(set_y)
    dist_vector = np.concatenate((set_x, set_y))
    mj1_distance = ((np.mean(set_x) + np.mean(set_y)) / 2)  # Reference 17
    return mj1_distance

def covariance_matrix(y, N_gen, x_grid, l_scale, var, noise_var):
    omega = np.zeros((N_gen, N_gen))
    for i in range(N_gen):
        for j in range(N_gen):
            omega[i, j] = var * np.exp(-1 / (2 * l_scale ** 2) * ((x_grid[i] - x_grid[j]) ** 2))

    # Generate Covariance matrix - predictive distribution
    k_xx = omega + np.eye(N_gen) * noise_var  # K(X,X)

    k_x_xstar = np.zeros((N_gen, len(x_grid)))  # K(X, X_star)
    for i in range(N_gen):
        for j in range(len(x_grid)):
            k_x_xstar[i, j] = var * np.exp(-1 / (2 * l_scale ** 2) * ((x_grid[i] - x_grid[j]) ** 2))

    k_xstar_x = np.zeros((len(x_grid), N_gen))  # K(X_star, X)
    for i in range(len(x_grid)):
        for j in range(N_gen):
            k_xstar_x[i, j] = var * np.exp(-1 / (2 * l_scale ** 2) * ((x_grid[i] - x_grid[j]) ** 2))

    k_xstar_xstar = np.zeros((len(x_grid), len(x_grid)))  # K(X_star, X_star)
    for i in range(len(x_grid)):
        for j in range(len(x_grid)):
            k_xstar_xstar[i, j] = var * np.exp(-1 / (2 * l_scale ** 2) * ((x_grid[i] - x_grid[j]) ** 2))

    f_star = np.dot(k_xstar_x, np.dot(np.linalg.inv(k_xx), y))
    cov_f_star = k_xstar_xstar - np.dot(np.dot(k_xstar_x, np.linalg.inv(k_xx)), k_x_xstar)

    return f_star, cov_f_star

def surge_locations(x, window, epsilon, min_low):
    N = 0
    idx = []
    max_idx = []
    min_idx = []

    for i in range(window, len(x) - window):
        left = x[(i-window):i]
        right = x[(i+1):(i + window)]
        current = x[i]
        if np.max(left) < current and current > np.max(right) and current > epsilon: # Maximum
            idx.append(i)
            max_idx.append(i)
            N += 1

    for i in range(window, len(x) - window):
        left = x[(i-window):i]
        right = x[(i+1):(i + window)]
        current = x[i]
        if len(max_idx) > 0:
            if np.min(left) > current and current < np.min(right) and i > max_idx[0]: # Maximum
                idx.append(i)
                min_idx.append(i)
                N += 1
        else:
            if np.min(left) > current and current < np.min(right) and current > min_low: # Maximum
                idx.append(i)
                min_idx.append(i)
                N += 1

    # Grid for final window
    xs = np.linspace(0,window,window)
    final_grid = x[(len(x) - window):len(x)]
    m = best_fit_slope(xs, final_grid) # gradient of final segment
    if m > 0.05:
        idx_high = np.argmax(final_grid)
        idx.append(idx_high + len(x) - window)
        max_idx.append(idx_high + len(x) - window)
    if m < -0.05:
        idx_low = np.argmin(final_grid)
        idx.append(idx_low + len(x) - window)
        min_idx.append(idx_low + len(x) - window)

    return idx, max_idx, min_idx, N

def estimate_k(distance_matrix):
    affinity = np.exp(-distance_matrix ** 2 / (2 * 1))

    L_norm = csgraph.laplacian(affinity, normed=True)
    n_components = L_norm.shape[0]

    # find the eigenvalues and eigenvectors
    vals, vecs = eigsh(L_norm, k=n_components, which="SM", sigma=1.0, maxiter=50000)
    min_clusters = 2
    max_clusters = 25
    idx_largest_gap = np.argmax(np.diff(vals[:max_clusters])) + 1
    k = max(min(max_clusters, idx_largest_gap), min_clusters)
    X = vecs[:, :k]
    Y = X / np.linalg.norm(X, ord=2, axis=1, keepdims=True)
    Y = np.nan_to_num(Y)

    # # K means clustering on k smallest eigenvectors of graph laplacian
    # kmeans = KMeans(n_clusters=k).fit(Y)
    # colors = kmeans.labels_
    # spectral_clust_labels.append(colors + 1)

    return k


def distcorr(X, Y):
    """ Compute the distance correlation function

    0.762676242417
    """
    X = np.atleast_1d(X)
    Y = np.atleast_1d(Y)
    if np.prod(X.shape) == len(X):
        X = X[:, None]
    if np.prod(Y.shape) == len(Y):
        Y = Y[:, None]
    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y)
    n = X.shape[0]
    if Y.shape[0] != X.shape[0]:
        raise ValueError('Number of samples must match')
    a = squareform(pdist(X))
    b = squareform(pdist(Y))
    A = a - a.mean(axis=0)[None, :] - a.mean(axis=1)[:, None] + a.mean()
    B = b - b.mean(axis=0)[None, :] - b.mean(axis=1)[:, None] + b.mean()

    dcov2_xy = (A * B).sum() / float(n * n)
    dcov2_xx = (A * A).sum() / float(n * n)
    dcov2_yy = (B * B).sum() / float(n * n)
    dcor = np.sqrt(dcov2_xy) / np.sqrt(np.sqrt(dcov2_xx) * np.sqrt(dcov2_yy))
    return dcor

def kernel_density_estimate(array):
    kde = gaussian_kde(array)
    dist_space = np.linspace(np.min(array), np.max(array), len(array))
    return dist_space, kde(dist_space) # dist space and kde

def time_varying_offset_function(new_cases, new_deaths, rolling_window, max_offset):
    time_varying_offset = []
    slice_inc = 0

    while len(time_varying_offset) < len(new_cases[0])-rolling_window: # len(cases[0])-rolling_window

        # Slices of cases and deaths
        cases_slice = np.nan_to_num(new_cases[:, slice_inc:(slice_inc+rolling_window)])
        deaths_slice = np.nan_to_num(new_deaths[:, slice_inc:(slice_inc+rolling_window)])

        print("length cases slice", len(cases_slice))
        print("length deaths slice", len(deaths_slice))
        print("Slice increment", slice_inc, slice_inc+rolling_window)

        # Optimise gap between cases and deaths
        system_deviation = []
        counter = 0

        while counter < max_offset:
            system_current_iteration = []
            for i in range(counter, len(cases_slice[0])):
                log_cases = manhattan_distances(np.log(np.where(cases_slice[:, i - counter]<=1,1,cases_slice[:, i - counter]).reshape(len(cases_slice[:,i - counter]), 1), np.log(np.where(cases_slice[:, i - counter]<=1,1,cases_slice[:, i - counter]).reshape(len(cases_slice[:, i - counter]), 1))))
                affinity_cases = 1 - np.nan_to_num((log_cases) / np.max(log_cases)) # sigma assumed to be one

                log_deaths = manhattan_distances(np.log(np.where(deaths_slice[:, i]<=1,1,deaths_slice[:, i]).reshape(len(deaths_slice[:, i]), 1), np.log(np.where(deaths_slice[:, i]<=1,1,deaths_slice[:, i]).reshape(len(deaths_slice[:, i]), 1))))
                affinity_deaths = 1 - np.nan_to_num((log_deaths) / np.max(log_deaths)) # sigma assumed to be one

                # adjacency_difference = np.linalg.norm(np.subtract(affinity_cases,affinity_deaths)) # l2 Norm
                affinity_difference = np.sum(np.abs(np.subtract(affinity_cases, affinity_deaths))) # l1 Norm
                system_current_iteration.append(affinity_difference)

            total_norm_difference = np.mean(system_current_iteration) # Should be a mean
            system_deviation.append(total_norm_difference)
            system_current_iteration.clear()
            counter += 1

        # Append time-varying offset value
        optimal_tvo = np.argmin(system_deviation) + 1
        time_varying_offset.append(optimal_tvo)
        print(optimal_tvo)

        # Increase in x for indicator
        print("system deviation list = ", len(system_deviation))

        # Increase increment on slice
        slice_inc += 1

    # plot time-varying offset
    plt.plot(time_varying_offset)
    plt.title("Time varying offset")
    plt.savefig("Time varying offset distribution "+str(rolling_window))
    plt.show()

    return time_varying_offset

# Time varying offset for PDFs
def time_varying_offset_pdfs(new_cases, new_deaths, rolling_window, max_offset):
    time_varying_offset = []
    slice_inc = 0

    while len(time_varying_offset) < len(new_cases[0]) - rolling_window:  # len(cases[0])-rolling_window

        # Slices of cases and deaths
        cases_slice = np.nan_to_num(new_cases[:, slice_inc:(slice_inc + rolling_window)])
        deaths_slice = np.nan_to_num(new_deaths[:, slice_inc:(slice_inc + rolling_window)])

        print("length cases pdf", len(cases_slice))
        print("length deaths pdf", len(deaths_slice))
        print("Slice increment", slice_inc, slice_inc + rolling_window)

        # Optimise gap between cases and deaths
        system_deviation = []
        counter = 0

        while counter < max_offset:
            system_current_iteration = []
            for i in range(counter, len(cases_slice[0])):
                cases_slice_sum = cases_slice[:,i-counter]
                cases_pdf = np.nan_to_num(cases_slice_sum / np.sum(cases_slice_sum))

                deaths_slice_sum = deaths_slice[:,i]
                deaths_pdf = np.nan_to_num(deaths_slice_sum / np.sum(deaths_slice_sum))

                pdf_difference = np.sum(np.abs(np.subtract(cases_pdf, deaths_pdf)))
                system_current_iteration.append(pdf_difference)

            total_norm_difference = np.mean(system_current_iteration)  # Should be a mean
            system_deviation.append(total_norm_difference)
            system_current_iteration.clear()
            counter += 1

        # Append time-varying offset value
        optimal_tvo = np.argmin(system_deviation) + 1
        time_varying_offset.append(optimal_tvo)
        print(optimal_tvo)

        # Increase in x for indicator
        print("system deviation list = ", len(system_deviation))

        # Increase increment on slice
        slice_inc += 1

    # plot time-varying offset
    plt.plot(time_varying_offset)
    plt.title("Time varying offset")
    plt.savefig("Time varying offset distribution " + str(rolling_window))
    plt.show()

    return time_varying_offset

# Time varying offset for Inner product
def time_varying_offset_inner_product(new_cases, new_deaths, rolling_window, max_offset):
    country_offsets = []
    time_varying_offset = []
    slice_inc = 0

    while len(time_varying_offset) < len(new_cases[0]) - rolling_window:  # len(cases[0])-rolling_window

        # Slices of cases and deaths
        cases_slice = np.nan_to_num(new_cases[:, slice_inc:(slice_inc + rolling_window)])
        deaths_slice = np.nan_to_num(new_deaths[:, slice_inc:(slice_inc + rolling_window)])

        print("length cases", len(cases_slice))
        print("length deaths", len(deaths_slice))
        print("Slice increment", slice_inc, slice_inc + rolling_window)

        # Optimise gap between cases and deaths
        optimal_offset_list = []
        counter = 0
        for i in range(len(cases_slice)): # Run over all the states
            inner_product_list = []
            for j in range(counter, max_offset): # Try offsets up to the maximum offset.
                if j == 0:
                    cases_iteration = cases_slice[i, (j-counter):]
                else:
                    cases_iteration = cases_slice[i, (j - counter):-j]
                deaths_iteration = deaths_slice[i, j:]
                inner_product = np.matmul(cases_iteration, deaths_iteration) * 1/len(cases_iteration)
                inner_product_list.append(inner_product)
                counter += 1
            # Append the argmax of the inner product list (optimal translation)
            opt_offset = np.argmax(inner_product_list)
            optimal_offset_list.append(opt_offset)
            counter = 0

        # Average TVO and append
        average_opt_tvo = np.mean(optimal_offset_list)
        time_varying_offset.append(average_opt_tvo)
        country_offsets.append(optimal_offset_list)
        # Increase in x for indicator
        print("system deviation list = ", len(optimal_offset_list))

        # Increase increment on slice
        slice_inc += 1

    # Country offset smoothing (only increase increment by one at a time)
    country_offsets_array = np.array(country_offsets) # Make an array
    smoothed_country_offsets = []
    for i in range(len(country_offsets_array[0])):
        country_slice = country_offsets_array[:,i]
        smoothed_iteration = []
        for j in range(len(country_slice)):
            if j == 0:
                smoothed_iteration.append(country_slice[j])
            else:
                if country_slice[j] > country_slice[j-1]:
                    smoothed_iteration.append(smoothed_iteration[-1]+1)
                if country_slice[j] < country_slice[j-1]:
                    smoothed_iteration.append(smoothed_iteration[-1]-1)
                if country_slice[j] == country_slice[j-1]:
                    smoothed_iteration.append(smoothed_iteration[-1])
        smoothed_country_offsets.append(smoothed_iteration) # Append smoothed iteration

    return time_varying_offset, smoothed_country_offsets

# Time varying offset for PDFs
def time_varying_offset_pdfs(new_cases, new_deaths, rolling_window, max_offset):
    time_varying_offset = []
    slice_inc = 0

    while len(time_varying_offset) < len(new_cases[0]) - rolling_window:  # len(cases[0])-rolling_window

        # Slices of cases and deaths
        cases_slice = np.nan_to_num(new_cases[:, slice_inc:(slice_inc + rolling_window)])
        deaths_slice = np.nan_to_num(new_deaths[:, slice_inc:(slice_inc + rolling_window)])

        print("length cases pdf", len(cases_slice))
        print("length deaths pdf", len(deaths_slice))
        print("Slice increment", slice_inc, slice_inc + rolling_window)

        # Optimise gap between cases and deaths
        system_deviation = []
        counter = 0

        while counter < max_offset:
            system_current_iteration = []
            for i in range(counter, len(cases_slice[0])):
                cases_slice_sum = cases_slice[:, i - counter]
                cases_pdf = np.nan_to_num(cases_slice_sum / np.sum(cases_slice_sum))

                deaths_slice_sum = deaths_slice[:, i]
                deaths_pdf = np.nan_to_num(deaths_slice_sum / np.sum(deaths_slice_sum))

                pdf_difference = np.sum(np.abs(np.subtract(cases_pdf, deaths_pdf)))
                system_current_iteration.append(pdf_difference)

            total_norm_difference = np.mean(system_current_iteration)  # Should be a mean
            system_deviation.append(total_norm_difference)
            system_current_iteration.clear()
            counter += 1

        # Append time-varying offset value
        optimal_tvo = np.argmin(system_deviation) + 1
        time_varying_offset.append(optimal_tvo)
        print(optimal_tvo)

        # Increase in x for indicator
        print("system deviation list = ", len(system_deviation))

        # Increase increment on slice
        slice_inc += 1

    # plot time-varying offset
    plt.plot(time_varying_offset)
    plt.title("Time varying offset")
    plt.savefig("Time varying offset distribution " + str(rolling_window))
    plt.show()

    return time_varying_offset

def drange(x, y, jump):
  while x < y:
    yield float(x)
    x += decimal.Decimal(jump)

# Time varying offset for Inner products
def time_varying_offset_inner_product(new_cases, new_deaths, rolling_window, max_offset):
    country_offsets = []
    time_varying_offset = []
    slice_inc = 0

    while len(time_varying_offset) < len(new_cases[0]) - rolling_window:  # len(cases[0])-rolling_window

        # Slices of cases and deaths
        cases_slice = np.nan_to_num(new_cases[:, slice_inc:(slice_inc + rolling_window)])
        deaths_slice = np.nan_to_num(new_deaths[:, slice_inc:(slice_inc + rolling_window)])

        print("length cases", len(cases_slice))
        print("length deaths", len(deaths_slice))
        print("Slice increment", slice_inc, slice_inc + rolling_window)

        # Optimise gap between cases and deaths
        optimal_offset_list = []
        counter = 0
        for i in range(len(cases_slice)): # Run over all the states
            inner_product_list = []
            for j in range(counter, max_offset): # Try offsets up to the maximum offset.
                if j == 0:
                    cases_iteration = cases_slice[i, (j-counter):]
                else:
                    cases_iteration = cases_slice[i, (j - counter):-j]
                deaths_iteration = deaths_slice[i, j:]
                inner_product = np.matmul(cases_iteration, deaths_iteration) * 1/len(cases_iteration)
                inner_product_list.append(inner_product)
                counter += 1
            # Append the argmax of the inner product list (optimal translation)
            opt_offset = np.argmax(inner_product_list)
            optimal_offset_list.append(opt_offset)
            counter = 0

        # Average TVO and append
        average_opt_tvo = np.mean(optimal_offset_list)
        time_varying_offset.append(average_opt_tvo)
        country_offsets.append(optimal_offset_list)
        # Increase in x for indicator
        print("system deviation list = ", len(optimal_offset_list))

        # Increase increment on slice
        slice_inc += 1

    # plot time-varying offset
    plt.plot(time_varying_offset)
    plt.title("Time varying offset")
    plt.savefig("Time varying offset distribution " + str(rolling_window))
    plt.show()

    return time_varying_offset, country_offsets

# Time varying offset for Inner product
def time_varying_offset_inner_product(new_cases, new_deaths, rolling_window, max_offset):
    country_offsets = []
    time_varying_offset = []
    slice_inc = 0

    while len(time_varying_offset) < len(new_cases[0]) - rolling_window:  # len(cases[0])-rolling_window

        # Slices of cases and deaths
        cases_slice = np.nan_to_num(new_cases[:, slice_inc:(slice_inc + rolling_window)])
        deaths_slice = np.nan_to_num(new_deaths[:, slice_inc:(slice_inc + rolling_window)])

        print("length cases", len(cases_slice))
        print("length deaths", len(deaths_slice))
        print("Slice increment", slice_inc, slice_inc + rolling_window)

        # Optimise gap between cases and deaths
        optimal_offset_list = []
        counter = 0
        for i in range(len(cases_slice)): # Run over all the states
            inner_product_list = []
            for j in range(counter, max_offset): # Try offsets up to the maximum offset.
                if j == 0:
                    cases_iteration = cases_slice[i, (j-counter):]
                else:
                    cases_iteration = cases_slice[i, (j - counter):-j]
                deaths_iteration = deaths_slice[i, j:]
                numerator = np.dot(cases_iteration, deaths_iteration)
                denominator = np.linalg.norm(cases_iteration) * np.linalg.norm(deaths_iteration)
                inner_product = numerator * 1/len(cases_iteration)
                inner_product_list.append(inner_product)
                counter += 1
            # Append the argmax of the inner product list (optimal translation)
            opt_offset = np.argmax(inner_product_list)
            optimal_offset_list.append(opt_offset)
            counter = 0

        # Average TVO and append
        average_opt_tvo = np.mean(optimal_offset_list)
        time_varying_offset.append(average_opt_tvo)
        country_offsets.append(optimal_offset_list)

        # Increase in x for indicator
        print("system deviation list = ", len(optimal_offset_list))

        # Increase increment on slice
        slice_inc += 1

    return time_varying_offset, country_offsets

def mj1_distance(x,y):
    set_x = []
    set_y = []
    for i in range(len(x)):
        set_x.append(np.abs(x[i] - find_nearest(y, x[i])))
    for j in range(len(y)):
        set_y.append(np.abs(y[j] - find_nearest(x, y[j])))
    set_x = np.array(set_x)
    set_y = np.array(set_y)
    dist_vector = np.concatenate((set_x, set_y))
    mj1_distance = ((np.mean(set_x) + np.mean(set_y)) / 2)  # Reference 17
    return mj1_distance

def sequential_norm_computation(cases, names):
    usa_cases_curr = cases
    names_new = names

    # Norm scores
    norm_scores = []
    names_list = []
    while len(norm_scores) < len(cases)-1:
        # Number of elements in current matrix
        num_elements = len(usa_cases_curr)
        norm = 1/(num_elements * (num_elements-1)) * np.sum(np.abs(usa_cases_curr)) # L1 norm of matrix x number of non-zero elements
        norm_scores.append(norm) # Append to empty list

        # Removal
        anomaly_scores = []
        for i in range(len(usa_cases_curr[0])):
            anomaly = np.sum(np.abs(usa_cases_curr[:,i]))
            anomaly_scores.append(anomaly)

        # Rank all elements in the list
        ranks = np.argsort(anomaly_scores)

        # Remove the element corresponding to the largest anomaly (note - indexes start at 1, not 0)
        largest_element = np.int(ranks[-1])

        # Append current state to anomaly order list
        names_list.append(names_new[largest_element]) # Append to names list

        # Update Current distance matrix by removing respective row and column
        usa_cases_curr = np.delete(usa_cases_curr, largest_element, axis=0)
        usa_cases_curr = np.delete(usa_cases_curr, largest_element, axis=1)

        # Remove name from list
        names_new.pop(largest_element)

    return norm_scores, names_list