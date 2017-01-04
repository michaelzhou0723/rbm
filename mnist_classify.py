from rbm import BinaryRBM
from mnist import load_MNIST, visualise_MNIST
import numpy as np

np.seterr(all='raise')

train_imgs, train_lbls = load_MNIST()
test_imgs, test_lbls = load_MNIST('test')
model = BinaryRBM(500, epochs=1, batch_size=10, learning_rate=0.1, cdk=1, persistent=True, measure='pl')
model.train(train_imgs / 255, train_lbls)
lbls = model.classify(test_imgs / 255)
print('Classification Error: {}'.format(np.sum(np.not_equal(lbls, test_lbls)) / lbls.shape[0]))
