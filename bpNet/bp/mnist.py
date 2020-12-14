import os
import struct
import numpy as np
import matplotlib.pyplot as plt

path = os.path.join(os.path.abspath(os.getcwd()), 'data')


class mnist:
    def __init__(self,
                 kind="train",
                 normalize=True,
                 flatten=True,
                 oneHotLabel=True):

        self.kind = kind  # Test data: kind="t10k"
        self.path = path
        print("Strat loading data...")

        if (normalize & flatten):
            # set self.images & self.labels
            self.load_mnist(self.path)

        if (oneHotLabel):
            self.oneHotLabels = self.changeOneHotLabel(self.labels)

    def load_mnist(self, path):

        labels_path = os.path.join(path, '%s-labels-idx1-ubyte' % self.kind)
        images_path = os.path.join(path, '%s-images-idx3-ubyte' % self.kind)

        with open(labels_path, 'rb') as lbpath:
            magic, n = struct.unpack('>II', lbpath.read(8))
            labels = np.fromfile(lbpath, dtype=np.uint8)

        with open(images_path, 'rb') as imgpath:
            magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
            images = np.fromfile(imgpath,
                                 dtype=np.uint8).reshape(len(labels), 784)

        print("save origin image as showImg...")
        self.showImg = images.reshape(images.shape[0], 28, 28)
        newData = images.astype(np.float)
        print("normalize image...")
        self.images = newData / 255.0
        self.labels = labels
        print("Data loading finished...")

    def changeOneHotLabel(self, data):

        newData = np.zeros((data.size, 10))
        for index, row in enumerate(newData):
            row[data[index]] = 1
        return newData

    def plot_img(self, idx):
        # showImage is a set of 28*28 grayscale images
        # The value of each element of the array is between 0 and 255 0 means white 255 means black
        fig = plt.gcf()
        print("set image config...")
        fig.set_size_inches(5, 5)
        print("imshow...")
        plt.imshow(self.showImg[idx], cmap="binary")
        print("image show...")
        plt.show()
        print("label=", self.labels[idx])
