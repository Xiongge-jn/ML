import numpy as np
import matplotlib.pyplot as plt
import gzip
# 训练集文件
train_images = 'MNIST_data/train-images-idx3-ubyte.gz'
# 训练集标签文件
train_labels = 'MNIST_data/train-labels-idx1-ubyte.gz'

# 测试集文件
test_images = 'MNIST_data/t10k-images-idx3-ubyte.gz'
# 测试集标签文件
test_labels = 'MNIST_data/t10k-labels-idx1-ubyte.gz'
IMAGE_SIZE = 28
NUM_CHANNELS = 1
PIXEL_DEPTH = 255
NUM_LABELS = 10
# Extract the images
def signmoid(z):
    s=1/(1+np.exp(-z))
    return s
def softmax(x):
    x_exp = np.exp(x)
    #如果是列向量，则axis=0
    x_sum = np.sum(x_exp, axis = 0, keepdims = True)
    s = x_exp / x_sum
    return s

def grant_sig(z):
    return signmoid(z) * (1 - signmoid(z))
def relu_forward(Z):
    """
    :param Z: Output of the activation layer
    :return:
    A: output of activation
    """
    A = np.maximum(0,Z)
    return A
def line_forward(x, w, b):
    z = np.dot(w, x) + b
    return z
def forward_propagation(X, parameters):
    #L = len(parameters) // 2
    L=2
    A = X
    caches = []
    for l in range(1, L):
        W = parameters["w" + str(l)]
        B = parameters["b" + str(l)]
        z = line_forward(A, W, B)
        caches.append((A, W, B, z))
        A = relu_forward(z)
    WL = parameters["w" + str(L)]
    bL = parameters["b" + str(L)]
    zL = line_forward(A, WL, bL)
    caches.append((A, WL, bL, zL))

    AL = softmax(zL)
    return AL, caches
def extract_data(filename, num_images):
    """Extract the images into a 4D tensor [image index, y, x, channels].
    Values are rescaled from [0, 255] down to [-0.5, 0.5].
    """
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(IMAGE_SIZE * IMAGE_SIZE * num_images * NUM_CHANNELS)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = (data - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH
        data = data.reshape(num_images, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)
        data = np.reshape(data, [num_images, -1])
    return data
def extract_labels(filename, num_images):
    """Extract the labels into a vector of int64 label IDs."""
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * num_images)
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
        num_labels_data = len(labels)
        one_hot_encoding = np.zeros((num_labels_data,NUM_LABELS))
        one_hot_encoding[np.arange(num_labels_data),labels] = 1
        one_hot_encoding = np.reshape(one_hot_encoding, [-1, NUM_LABELS])
    return one_hot_encoding
train_data = extract_data(train_images, 60000)
train_labels = extract_labels(train_labels, 60000)
test_data = extract_data(test_images, 10000)
test_labels = extract_labels(test_labels, 10000)


def init_w_b(layer_dims):
    np.random.seed(16)
    L = len(layer_dims)
    parameters = {}
    for l in range(1, L):
        parameters["w" + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * 0.1
        parameters["b" + str(l)] = np.zeros((layer_dims[l], 1))
    return parameters

sets_number=40
plt.figure()

paras=np.load("test.npy",allow_pickle=True)
paras=paras.item()




testcmm=test_data[:32]

yww=np.argmax(test_labels,axis=1)
cmm=yww[:32].reshape(4,8)
print("labels:\n")
print(cmm)
AL, caches=forward_propagation(testcmm.T,paras)
xww=np.argmax(AL,axis=0)
xmm=xww[:32].reshape(4,8)
print("predict:\n")
print(xmm)
for i in range(1,33):

    plt.subplot(4,8,i)
    img = test_data[i-1].reshape(28, 28)




    plt.imshow(img)

plt.show()
