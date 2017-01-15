import struct
import numpy as np
import matplotlib.pyplot as plt

def load_MNIST(dataset = 'train'):
    if dataset == 'train':
        images_path = './train-images.idx3-ubyte'
        labels_path = './train-labels.idx1-ubyte'
    elif dataset == 'test':
        images_path = './t10k-images.idx3-ubyte'
        labels_path = './t10k-labels.idx1-ubyte'
    else:
        raise ValueError
    dt = np.dtype(np.uint8)
    dt.newbyteorder('>')       
    with open(labels_path, 'rb') as f:
        magic_num, num_labels = struct.unpack('>II', f.read(8))
        labels = np.frombuffer(f.read(), dt, num_labels)
    with open(images_path, 'rb') as f:
        magic_num, num_images, num_rows, num_cols = struct.unpack('>IIII', f.read(16))
        images = np.frombuffer(f.read(), dt, num_images * num_rows * num_cols)
        images = images.reshape(num_images, num_rows*num_cols)
    return images, labels
    
def visualise_MNIST(pixels, ax = None):
    pixels = np.array(pixels, dtype='uint8')
    pixels = pixels.reshape((28, 28))
    if ax is not None:
        ax.imshow(pixels, cmap='gray')
    else:
        plt.imshow(pixels, cmap='gray')
        plt.show()
    

