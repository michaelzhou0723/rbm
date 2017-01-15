import os, sys
sys.path.append(os.path.abspath('../'))
from rbm import BinaryRBM
from utils import load_model
from mnist import load_MNIST, visualise_MNIST
import numpy as np
import matplotlib.pyplot as plt

test_imgs, test_lbls = load_MNIST('test')
rand_ind = np.random.randint(0, test_imgs.shape[0] - 1, (100,))

model = load_model()
recon_data = model.reconstruct(test_imgs[rand_ind, :], test_lbls[rand_ind])

fig, axes = plt.subplots(10, 10, subplot_kw={'xticks': [], 'yticks': []})
for i, ax in enumerate(axes.flat):
    visualise_MNIST(recon_data[i] * 255, ax)
    
plt.suptitle('Reconstructed MNIST figures', size=26)
plt.show()



