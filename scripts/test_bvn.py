from voi.birkoff_utils import birkhoff_von_neumann
from voi.nn.base.stick_breaking import stick_breaking, inv_stick_breaking
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":

    xs = [10000, 1000, 100, 10, 1, 0.1, 0.01, 0.001, 0.0001]
    ys = [[], [], [], [], [], [], [], [], []]

    for seed in range(3):

        normal = tf.random.normal([1, 12, 12])

        for pos, temp in enumerate(xs):

            logits = normal / temp
            mask = tf.ones_like(logits)

            def normalize(inner_logits):
                """Reconstruct a doubly stochastic matrix using the
                Birkhoff-Von Neuman Decomposition and Greedy Birkhoff Heuristic"""

                x = mask / tf.reduce_sum(mask, axis=1, keepdims=True)
                z = inv_stick_breaking(x, mask)

                a = stick_breaking(tf.math.sigmoid(inner_logits -
                                                   tf.math.log(1. / z - 1.)), mask)

                return a

            def reconstruct(inner_logits):
                """Reconstruct a doubly stochastic matrix using the
                Birkhoff-Von Neuman Decomposition and Greedy Birkhoff Heuristic"""

                x = mask / tf.reduce_sum(mask, axis=1, keepdims=True)
                z = inv_stick_breaking(x, mask)

                a = stick_breaking(tf.math.sigmoid(inner_logits -
                                                   tf.math.log(1. / z - 1.)), mask)

                p, z = birkhoff_von_neumann(a, 1000)

                return tf.reduce_sum(
                    p * z[:, :, tf.newaxis, tf.newaxis], axis=1)

            theoretical, numeric = tf.test.compute_gradient(reconstruct,
                                                            [logits],
                                                            delta=0.001)
            theoretical2, numeric2 = tf.test.compute_gradient(normalize,
                                                              [logits],
                                                              delta=0.001)
            cumulative_error = 0.0

            # if nothing is printed, then the theoretical gradients agree
            for t, n, t2, n2 in zip(theoretical, numeric, theoretical2, numeric2):
                for i, (x0, x1, y0, y1) in enumerate(zip(t.flat, n.flat, t2.flat, n2.flat)):
                    cumulative_error += abs(x0 - y0)

            ys[pos].append(cumulative_error)
            print(pos, temp, cumulative_error)

    # calculate statistics
    mean = np.percentile(ys, 50, axis=1)
    uppr = np.percentile(ys, 25, axis=1)
    lowr = np.percentile(ys, 75, axis=1)

    # plot the gradient error in a figure
    ax = plt.subplot(111)
    plt.plot(xs, mean, color=(0.8, 0.2, 0.0, 1.0))
    plt.fill_between(xs, uppr, lowr, color=(0.8, 0.2, 0.0, 0.2))
    plt.xscale("log")
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plt.title("BvN Gradient Error")
    plt.xlabel("Temperature")
    plt.ylabel("Gradient Error L1 Norm")
    plt.savefig("gradient_error.png")
