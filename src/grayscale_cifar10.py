from keras.datasets import cifar10
import numpy as np
import pickle
from PIL import Image

def grayscale(pixel):
    """
    Input is a single pixel. Output is a weighted average of red, green and blue.
    """
    return 0.299 * pixel[0] + 0.587 * pixel[1] + 0.114 * pixel[2]

def iter_over_image(image):
    """
    Iterate over each pixel in image, pass each pixel to grayscale function
    """
    for i in range(len(image)):
        for j,pixel in enumerate(image[i]):
            image[i][j] = grayscale(pixel)
    return image


if __name__ == '__main__':
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    #8 is for ships
    X_train = [X_train[i] for i in range(len(y_train)) if y_train[i] == 8]
    X_test = [X_test[i] for i in range(len(y_test)) if y_test[i] == 8]

    X_train_true = np.array(X_train)
    # y_train_img = Image.fromarray(y_train[0],'RGB')
    # y_train_img.show()
    with open('../data/X_train_true.p','wb') as f:
        pickle.dump(X_train_true,f)
    print('X_train_true done...')

    X_train = np.array([iter_over_image(image) for image in X_train])
    # X_train_img = Image.fromarray(X_train[0],'RGB')
    # X_train_img.show()
    with open('../data/X_train.p','wb') as f:
        pickle.dump(X_train,f)
    print('X_train done...')

    X_test_true = np.array(X_test)
    # y_test_img = Image.fromarray(y_test[0],'RGB')
    # y_test_img.show()
    with open('../data/X_test_true.p','wb') as f:
        pickle.dump(X_test_true,f)
    print('X_test_true done...')


    X_test = np.array([iter_over_image(image) for image in X_test])
    # X_test_img = Image.fromarray(X_test[0],'RGB')
    # X_test_img.show()
    with open('../data/X_test.p','wb') as f:
        pickle.dump(X_test,f)
    print('X_test done...')
