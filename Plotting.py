import matplotlib.pyplot as plt
import numpy as np

first_layer_avg_biases = \
    [-0.0004902551, 0.00039006956, 0.0017197693, 0.0030884123,
     0.004438256, 0.0059728604, 0.007015775, 0.0081689535,
     0.008876735, 0.009865081, 0.010583663, 0.011443974,
     0.011922095, 0.0123446, 0.012621021, 0.013127385,
     0.013087186, 0.01332649, 0.013686702, 0.013945067,
     0.014367754, 0.014558991, 0.014862154, 0.015050579,
     0.015483868, 0.01559619, 0.015689708]
seventh_layer_avg_biases = \
    [-0.006086599, -0.007286721, -0.0083071375, -0.009332911,
     -0.010105677, -0.010805508, -0.01132076, -0.0116214845,
     -0.012080483, -0.012335246, -0.012659961, -0.012891183,
     -0.013144494, -0.013432878, -0.013462259, -0.0138339335,
     -0.013731648, -0.014006524, -0.014126557, -0.01429073,
     -0.014381699, -0.014434192, -0.01454813, -0.014571531,
     -0.014740968, -0.014979419, -0.015015159]

first_layer_avg_weights = \
    [2.45740721e-05, 8.97260234e-05, 1.98221765e-04, 2.22489500e-04, 5.00722963e-05, 6.65484113e-05,
     2.46367999e-05, 6.41902443e-06, 1.94698805e-05, 2.84348498e-05, 3.14977951e-05, 3.53928190e-05,
     2.65159179e-06, 2.70836754e-05, 1.59621122e-05, 5.65563096e-05, 8.69051437e-05, 2.77019572e-05,
     1.20724435e-05, 1.94617314e-05, 3.95660754e-06, 1.21808262e-05, 8.77649290e-06, 9.24570486e-06,
     4.32784436e-05, 9.92976129e-06, 1.23269856e-05, 8.02174094e-04]
seventh_layer_avg_weights = \
    [1.48216568e-04, 4.43615965e-04, 4.79131180e-04, 5.17537468e-04,
     3.99899902e-04, 3.24574299e-04, 1.62945362e-04, 1.57547882e-04,
     2.85056885e-04, 1.76555244e-04, 1.73990615e-04, 3.06051224e-05,
     8.83759931e-05, 1.24943908e-04, 2.30164733e-05, 1.58766983e-04,
     6.66233245e-05, 2.53689941e-05, 1.50164589e-04, 1.00536970e-04,
     1.05922809e-04, 6.44382089e-06, 1.40618533e-04, 8.18041153e-05,
     1.26789790e-04, 2.78986990e-05, 6.77192584e-05, 3.95064475e-03]

# Find changes in the biases/weights
kernel = [-1, 1]
first_layer_avg_biases = np.abs(np.convolve(first_layer_avg_biases, kernel))
seventh_layer_avg_biases = np.abs(np.convolve(seventh_layer_avg_biases, kernel))
first_layer_avg_weights = np.abs(np.convolve(first_layer_avg_weights, kernel))
seventh_layer_avg_weights = np.abs(np.convolve(seventh_layer_avg_weights, kernel))

x = []
for i in range(len(first_layer_avg_weights)-1):
    x.append(i+1)

plt.subplot(2, 1, 1)
plt.tight_layout(pad=5.0)
plt.plot(x, first_layer_avg_biases, label='Layer 1')
plt.plot(x, seventh_layer_avg_biases, label='Layer 7')
plt.title('Biases')
plt.xlabel('Training Batches')
plt.ylabel('Value')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(x, first_layer_avg_weights[:-1], label='Layer 1')
plt.plot(x, seventh_layer_avg_weights[:-1], label='Layer 7')
plt.title('Weights')
plt.xlabel('Training Batches')
plt.ylabel('Change')
plt.legend()

plt.savefig('Change_in_weights.png')
plt.show()