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
    # new_img = np.ones(image.shape)
    # for row in image:
    #     for col in image:
    #         pixel = image[row,col]
    #         new_img[row,col] = [pixel[0] * .299, pixel[1] * .587, pixel[2] * .114]

def un_scale(image):
    image = np.squeeze(image)
    image = image * 100
    return image

def rgb_to_lab(image, l=False, ab=False):
    lab = color.rgb2lab(image)
    if l: l_layer = np.zeros((32,32,1))
    else: ab_layers = np.zeros((32,32,2))
    for i in range(len(lab)):
        for j in range(len(lab[i])):
            p = lab[i,j]
            # new_img[i,j] = [p[0]/100,(p[1] + 128)/255,(p[2] + 128)/255]
            if ab: ab_layers[i,j] = [(p[1] + 127)/255 * 2 - 1,(p[2] + 128)/255 * 2 -1]
            else: l_layer[i,j] = [p[0]/50 - 1]
    if l: return l_layer
    else: return ab_layers

def lab_to_rgb(image):
    new_img = np.zeros((32,32,3))
    for i in range(len(image)):
        for j in range(len(image[i])):
            p = image[i,j]
            new_img[i,j] = [(p[0] + 1) * 50,(p[1] +1) / 2 * 255 - 127,(p[2] +1) / 2 * 255 - 128]
    new_img = color.lab2rgb(new_img) * 255
    new_img = new_img.astype('uint8')
    return new_img

if __name__ == '__main__':
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    #8 is for ships
    X_train = np.array([X_train[i] for i in range(len(y_train)) if y_train[i] == 8])
    X_test = np.array([X_test[i] for i in range(len(y_test)) if y_test[i] == 8])

    X_train_L = np.array([rgb_to_lab(image, l=True) for image in X_train])
    print('X_train L layer done...')
    X_train_AB = np.array([rgb_to_lab(image, ab=True) for image in X_train])
    print('X_train a*b* layers done...')
    X_train = (X_train_L, X_train_AB)
    with open('../data/X_train.p','wb') as f:
        pickle.dump(X_train,f)
    print('X_train done...')

    X_test_L = np.array([rgb_to_lab(image,l=True) for image in X_test])
    print('X_test L layer done...')
    X_test_AB = np.array([rgb_to_lab(image, ab=True) for image in X_test])
    print('X_test a*b* layers done...')
    X_test = (X_test_L, X_test_AB)
    with open('../data/X_test.p','wb') as f:
        pickle.dump(X_test,f)
    print('X_test done...')

    # X_train_true = X_train
    # X_train_true = np.array([rgb_to_lab(image)[0] for image in X_train_true])
    # # X_train_true_img = lab_to_rgb(X_train_true[0])
    # # X_train_true_img = Image.fromarray(X_train_true_img,'RGB')
    # # X_train_true_img.show()
    # with open('../data/X_train_true.p','wb') as f:
    #     pickle.dump(X_train_true,f)
    # print('X_train_true done...')
    #
    # X_train = np.array([grayscale_image(image) for image in X_train])
    # # X_train_img = un_scale(X_train[0])
    # # X_train_img = Image.fromarray(X_train_img,'L')
    # # X_train_img.show()
    # with open('../data/X_train.p','wb') as f:
    #     pickle.dump(X_train,f)
    # print('X_train done...')
    #
    # X_test_true = np.array(X_test)
    # X_test_true = np.array([rgb_to_lab(image) for image in X_test_true])
    # # X_test_true_img = lab_to_rgb(X_test_true[0])
    # # X_test_true_img = Image.fromarray(X_test_true_img,'RGB')
    # # X_test_true_img.show()
    # with open('../data/X_test_true.p','wb') as f:
    #     pickle.dump(X_test_true,f)
    # print('X_test_true done...')
    #
    #
    # X_test = np.array([grayscale_image(image) for image in X_test])
    # # X_test_img = Image.fromarray(X_test[0],'L')
    # # X_test_img.show()
    # with open('../data/X_test.p','wb') as f:
    #     pickle.dump(X_test,f)
    # print('X_test done...')
