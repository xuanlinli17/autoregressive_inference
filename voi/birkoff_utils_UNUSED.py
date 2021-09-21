from networkx.algorithms.bipartite.matching import maximum_matching
from networkx import from_numpy_matrix
import tensorflow as tf
import numpy as np


TOLERANCE = np.finfo(np.float).eps * 10.


def find_largest_t(valid_matches, weights, edge_matrix):
    """Calculates the largest edge threshold using binary search such that
    the found maximum cardinality matching is a perfect matching

    Arguments:

    valid_matches: dict
        the last found perfect matching of largest edge weight threshold
        found using a binary search
    weights: np.ndarray
        an array of candidate thresholds for the edge weights in sorted order,
        where the largest elements are first
    edge_matrix: tf.Tensor
        a matrix of edge weights that correspond to a bipartite graph; used for
        constructing a bipartite graph in networkx

    Returns:

    permutation: tf.Tensor
        a tensor containing the Berkhoff-Von-Neumann perumtation matrices
        found using the Berkhoff-Von-Neumann decomposition"""

    # calculate the current loc of binary search
    n, loc = edge_matrix.shape[1] // 2, (weights.size - 1) // 2

    # calculate the bipartite graph whose edges all have weight of at
    # least the largest threshold found so far
    threshold = weights[loc]
    bipartite_matrix = np.where(edge_matrix >= threshold, 1, 0)

    # calculate the maximum matching using the hopcroft karp algorithm
    matches = maximum_matching(from_numpy_matrix(bipartite_matrix), range(n))
    matches = {u: v % n for u, v in matches.items() if u < n}

    # calculate if the found matching is a perfect matching
    is_perfect_matching = len(matches) == n
    valid_matches = matches if is_perfect_matching else valid_matches

    # otherwise if the result found is a perfect matching
    # then move onto larger thresholds
    if weights.size > 2 and is_perfect_matching:
        return find_largest_t(valid_matches, weights[:loc], edge_matrix)

    # otherwise if the result found is not a perfect matching
    # then move onto smaller thresholds
    elif weights.size > 1 and not is_perfect_matching:
        return find_largest_t(valid_matches, weights[loc + 1:], edge_matrix)

    # edge case when no valid permutation is a perfect matching and
    # the decomposition terminates with coefficient zero
    if not valid_matches:
        return np.ones((n, n), dtype=np.float32)

    # at the last iteration of binary search return the best
    # permutation matrix found so far
    permutation = np.zeros((n, n), dtype=np.float32)
    permutation[tuple(zip(*valid_matches.items()))] = 1
    return permutation


def get_permutation_np(edge_matrix):
    """Calculates the largest edge threshold using binary search such that
    the found maximum cardinality matching is a perfect matching

    Arguments:

    edge_matrix: tf.Tensor
        a matrix of edge weights that correspond to a bipartite graph; used for
        constructing a bipartite graph in networkx

    Returns:

    permutation: tf.Tensor
        a tensor containing the Berkhoff-Von-Neumann perumtation matrices
        found using the Berkhoff-Von-Neumann decomposition"""

    # obtain a sorted list of edge weights to perform a binary search over
    # to find the large edge weight threshold such that a perfect matching
    # exists in the graph with edges of weight greater than t
    # https://cstheory.stackexchange.com/questions/32321/
    #     weighted-matching-algorithm-for-minimizing-max-weight
    n = edge_matrix.shape[1] // 2
    weights = np.sort(edge_matrix[np.nonzero(edge_matrix)])[::-1]
    return np.ones((n, n), dtype=np.float32) if weights.size == 0 \
        else find_largest_t({}, weights, edge_matrix)


def get_permutation(edge_matrix):
    """Calculates the maximum cardinality perfect matching using networkx
    that corresponds to a permutation matrix

    Arguments:

    edge_matrix: tf.Tensor
        a binary matrix that corresponds to a bipartite graph; used for
        constructing a bipartite graph in networkx

    Returns:

    permutation: tf.Tensor
        a tensor containing the Berkhoff-Von-Neumann perumtation matrices
        found using the Berkhoff-Von-Neumann decomposition"""

    return tf.numpy_function(
        get_permutation_np, [edge_matrix], tf.float32)


def birkhoff_von_neumann_step(matrix):
    """Returns the Berkhoff-Von-Neumann decomposition of a permutation matrix
    using the greedy birkhoff heuristic

    Arguments:

    matrix: tf.Tensor
        a soft permutation matrix in the Birkhoff-Polytope whose shape is
        like [batch_dim, sequence_len, sequence_len]

    Returns:

    permutation: tf.Tensor
        a tensor containing the Berkhoff-Von-Neumann perumtation matrix
        found for the remaining values in matrix
    coefficient: tf.Tensor
        a tensor containing the Berkhoff-Von-Neumann coefficient
        found for the remaining values in matrix"""

    b, m, n = tf.shape(matrix)[0], tf.shape(matrix)[1], tf.shape(matrix)[2]

    # convert the matrix into an edge matrix of a bipartite graph
    top = tf.concat([tf.zeros([b, m, m]), matrix], axis=2)
    edge_matrix = tf.concat([top, tf.concat([tf.transpose(
        matrix, [0, 2, 1]), tf.zeros([b, n, n])], axis=2)], axis=1)

    # get a permutation matrix whose minimum edge weight is maximum
    permutation = tf.map_fn(get_permutation, edge_matrix)
    permutation.set_shape(matrix.get_shape())
    upper_bound = tf.fill(tf.shape(matrix), np.inf)

    return permutation, tf.reduce_min(tf.where(
        tf.equal(permutation, 0), upper_bound, matrix), axis=[1, 2])


# @tf.function(input_signature=[
#     tf.TensorSpec(shape=[None, None, None], dtype=tf.float32),
#     tf.TensorSpec(shape=None, dtype=tf.int32)])
def birkhoff_von_neumann(x, max_iterations):
    """Returns the Berkhoff-Von-Neumann decomposition of a permutation matrix
    using the greedy birkhoff heuristic

    Arguments:

    x: tf.Tensor
        a soft permutation matrix in the Birkhoff-Polytope whose shape is
        like [batch_dim, sequence_len, sequence_len]
    max_iterations: int
        the maximum number of matrices to compose to reconstruct
        the doubly stochastic matrix x

    Returns:

    permutations: tf.Tensor
        a tensor containing the Berkhoff-Von-Neumann perumtation matrices
        found using the Berkhoff-Von-Neumann decomposition
        shapes like [batch_dim, num_permutations, sequence_len, sequence_len]
    coefficients: tf.Tensor
        a tensor containing the Berkhoff-Von-Neumann coefficients
        found using the Berkhoff-Von-Neumann decomposition
        shapes like [batch_dim, num_permutations]"""

    b, n = tf.shape(x)[0], tf.cast(tf.shape(x)[2], tf.float32)
    x = x * n

    # keep track of a sequence of all permutations and coefficients
    coefficients = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    permutations = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

    j = tf.constant(-1)
    d = tf.reduce_all(tf.equal(x, 0), axis=[1, 2])

    # all permutations with coefficient 0 are set to the identity matrix
    eye_matrix = tf.eye(tf.shape(x)[2], batch_shape=[b])

    while tf.logical_and(tf.logical_not(
            tf.reduce_all(d)), tf.less(j + 1, max_iterations)):
        j = j + 1

        # compute the permutation matrix whose coefficient is maximum
        # we are done if the coefficient is zero
        p, c = birkhoff_von_neumann_step(x)
        d = tf.logical_or(d, tf.equal(c, tf.zeros_like(c)))

        # when we are done set the permutation to the identity matrix and
        # the coefficient to zero
        p = tf.where(d[:, tf.newaxis, tf.newaxis], eye_matrix, p)
        c = tf.where(d, tf.zeros_like(c), c)

        # iteratively subtract from the source matrix x until that matrix
        # is approximately zero everywhere
        x = x - c[:, tf.newaxis, tf.newaxis] * p
        x = tf.where(tf.less(tf.abs(x), TOLERANCE), tf.zeros_like(x), x)
        d = tf.logical_or(d, tf.reduce_all(tf.equal(x, 0), axis=[1, 2]))

        permutations = permutations.write(j, p)
        coefficients = coefficients.write(j, c)

    # the num_permutations axis is first and needs to be transposed
    return (tf.transpose(permutations.stack(), [1, 0, 2, 3]),
            tf.transpose(coefficients.stack(), [1, 0]) / n)
