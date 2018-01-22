import numpy as np
import pickle
from PIL import Image
from skimage import color, io
import matplotlib.pyplot as plt

def un_scale(image):
    image = np.squeeze(image)
    image = (image + 1) * 50
    return image

def rgb_to_lab(image, l=False, ab=False):
    lab = color.rgb2lab(image)
    if l: l_layer = np.zeros((32,32,1))
    else: ab_layers = np.zeros((32,32,2))
    for i in range(len(lab)):
        for j in range(len(lab[i])):
            p = lab[i,j]
            # new_img[i,j] = [p[0]/100,(p[1] + 128)/255,(p[2] + 128)/255]
            if ab: ab_layers[i,j] = [(p[1] + 128)/255 * 2 - 1,(p[2] + 128)/255 * 2 -1]
            else: l_layer[i,j] = [p[0]/50 - 1]
    if l: return l_layer.astype('uint8')
    else: return ab_layers.astype('uint8')

def lab_to_rgb(image):
    new_img = np.zeros((32,32,3))
    for i in range(len(image)):
        for j in range(len(image[i])):
            p = image[i,j]
            new_img[i,j] = [(p[0] + 1) * 50,(p[1] +1) / 2 * 255 - 128,(p[2] +1) / 2 * 255 - 128]
    new_img = color.lab2rgb(new_img) * 255
    new_img = new_img.astype('uint8')
    return new_img

if __name__ == '__main__':
    red = io.imread('../data/Paint/red.jpg')
    blue = io.imread('../data/Paint/blue.jpg')
    green = io.imread('../data/Paint/green.jpg')
    images = np.array([red,blue,green])
    X_train = np.array([images[i] for i in np.random.randint(0,3,500)])
    X_test = np.array([images[i] for i in np.random.randint(0,3,100)])

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

    # train = np.concatenate((rgb_to_lab(red,l=True),rgb_to_lab(red,ab=True)),axis=-1)
    # train = lab_to_rgb(train)
    # train_rgb = Image.fromarray(train,'RGB')
    # train_rgb.show()

    # train = rgb_to_lab(red,l=True)
    # train = un_scale(train)
    # train = Image.fromarray(train,'L')
    # train.show()
