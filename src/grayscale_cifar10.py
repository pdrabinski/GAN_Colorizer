from keras.datasets import cifar10
import numpy as np
import pickle
from PIL import Image
from skimage import color, io
import matplotlib.pyplot as plt

def grayscale_image(image):
    # image_file = Image.fromarray(image,'RGB')
    # arr = np.array(image_file.convert('L'))
    # arr = arr / 100
    arr = color.rgb2grey(image)
    return arr[...,np.newaxis]

def un_scale(image):
    image = np.squeeze(image)
    image = image * 100
    return image

def rgb_to_lab(image):
    lab = color.rgb2lab(image)
    new_img = np.zeros((32,32,3))
    for i in range(len(lab)):
        for j in range(len(lab[i])):
            p = lab[i,j]
            new_img[i,j] = [p[0]/100,(p[1] + 128)/255,(p[2] + 128)/255]
    return new_img

def lab_to_rgb(image):
    new_img = np.zeros((32,32,3))
    for i in range(len(image)):
        for j in range(len(image[i])):
            p = image[i,j]
            new_img[i,j] = [int(p[0] * 100),int(p[1] * 255 - 128),int(p[2] * 255 - 128)]
    new_img = color.lab2rgb(new_img) * 255
    new_img = new_img.astype('uint8')
    return new_img

if __name__ == '__main__':
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    #8 is for ships
    X_train = np.array([X_train[i] for i in range(len(y_train)) if y_train[i] == 8])
    X_test = np.array([X_test[i] for i in range(len(y_test)) if y_test[i] == 8])

    X_train_true = X_train
    # X_train_true_img = Image.fromarray(X_train_true[0],'RGB')
    # X_train_true_img.show()
    X_train_true = np.array([rgb_to_lab(image) for image in X_train_true])
    X_train_true_img = lab_to_rgb(X_train_true[0])
    X_train_true_img = Image.fromarray(X_train_true_img,'RGB')
    X_train_true_img.show()
    with open('../data/X_train_true.p','wb') as f:
        pickle.dump(X_train_true,f)
    print('X_train_true done...')

    X_train = np.array([grayscale_image(image) for image in X_train])
    X_train_img = un_scale(X_train[0])
    X_train_img = Image.fromarray(X_train_img,'L')
    X_train_img.show()
    with open('../data/X_train.p','wb') as f:
        pickle.dump(X_train,f)
    print('X_train done...')

    X_test_true = np.array(X_test)
    X_test_true = np.array([rgb_to_lab(image) for image in X_test_true])
    # X_test_true_img = lab_to_rgb(X_test_true[0])
    # X_test_true_img = Image.fromarray(X_test_true_img,'RGB')
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
