from keras.datasets import cifar10
import numpy as np
import pickle
from PIL import Image
from skimage import color

def grayscale_image(image):
    image_file = Image.fromarray(image,'RGB')
    arr = np.array(image_file.convert('L'))
    # arr = arr / 255 * 2 - 1
    return arr[...,np.newaxis]

def rgb_to_lab(image):
    lab = color.rgb2lab(image)
    return lab

if __name__ == '__main__':
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    #8 is for ships
    X_train = np.array([X_train[i] for i in range(len(y_train)) if y_train[i] == 8])
    X_test = np.array([X_test[i] for i in range(len(y_test)) if y_test[i] == 8])

    X_train_true = X_train
    X_train_true = np.array([rgb_to_lab(image) for image in X_train_true])
    # X_train_true_img = Image.fromarray(X_train_true[0],'RGB')
    # X_train_true_img.show()
    with open('../data/X_train_true.p','wb') as f:
        pickle.dump(X_train_true,f)
    print('X_train_true done...')

    X_train = np.array([grayscale_image(image) for image in X_train])
    # X_train_img = Image.fromarray(X_train[0],'L')
    # X_train_img.show()
    with open('../data/X_train.p','wb') as f:
        pickle.dump(X_train,f)
    print('X_train done...')

    X_test_true = np.array(X_test)
    X_test_true = np.array([rgb_to_lab(image) for image in X_test_true])
    # X_test_true_img = Image.fromarray(X_test_true[0],'RGB')
    # X_test_true_img.show()
    with open('../data/X_test_true.p','wb') as f:
        pickle.dump(X_test_true,f)
    print('X_test_true done...')


    X_test = np.array([grayscale_image(image) for image in X_test])
    # X_test_img = Image.fromarray(X_test[0],'L')
    # X_test_img.show()
    with open('../data/X_test.p','wb') as f:
        pickle.dump(X_test,f)
    print('X_test done...')
