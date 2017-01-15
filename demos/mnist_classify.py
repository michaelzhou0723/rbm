import os, sys
sys.path.append(os.path.abspath('../'))
from rbm import BinaryRBM
from mnist import load_MNIST
import numpy as np

train_imgs, train_lbls = load_MNIST()
test_imgs, test_lbls = load_MNIST('test')

model = BinaryRBM(1000, epochs=20, batch_size=50, learning_rate=0.05, weight_decay=0.001, momentum=0.5, cdk=1,
                  persistent=True, measure='pl', save=True)

model.train(train_imgs, train_lbls)
lbls = model.classify(test_imgs)

print('Classification Error: {:.2%}'.format(np.sum(np.not_equal(lbls, test_lbls)) / test_lbls.shape[0]))
