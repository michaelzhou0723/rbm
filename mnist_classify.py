from rbm import BinaryRBM
from mnist import load_MNIST, visualise_MNIST
import numpy as np

np.seterr(all='raise')

train_imgs, train_lbls = load_MNIST()
test_imgs, test_lbls = load_MNIST('test')
model = BinaryRBM(500)
model.feed(train_imgs / 255, train_lbls)
model.train(epochs=20, batch_size=100, lrate=0.1, cdk=100, persistent=False)
count = 0
for i in range(10000):
    lbl = model.classify(test_imgs[i] / 255, cdk=1)
    if lbl != test_lbls[i]:
        count += 1
print('Classification Error: {}'.format(count / 10000))
