from rbm import BinaryRBM
from mnist import load_MNIST, visualise_MNIST

train_imgs, train_lbls = load_MNIST()
model = BinaryRBM(train_imgs.shape[1], 1000)
model.train(train_imgs / 255, train_lbls, epochs=20, batch_size=100)
pixels = model.daydream(8, 10)
visualise_MNIST(pixels)

