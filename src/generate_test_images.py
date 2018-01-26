from keras.models import load_model
import pickle
import numpy as np
from PIL import Image
import time
import os
from skimage import color

np.random.seed(1)

def load_images(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def grayscale_image(images):
    """
    Grayscale image. Not used.
    """
    image = color.rgb2grey(images)
    image = (image * 255).astype('uint8')
    return image

def lab_to_rgb(l_layer, ab_layers, img_size):
    new_img = np.zeros((img_size,img_size,2))
    rescaled_l = np.zeros((img_size,img_size,1))
    for i in range(len(ab_layers)):
        for j in range(len(ab_layers[i])):
            p = ab_layers[i,j]
            new_img[i,j] = [(p[0] +1) / 2 * 255 - 128,(p[1] +1) / 2 * 255 - 128]
            rescaled_l[i,j] = [(l_layer[i,j] + 1) * 50]

    # print(rescaled_l.shape)
    # print(new_img.shape)
    new_img = np.concatenate((rescaled_l,new_img),axis=-1)
    new_img = color.lab2rgb(new_img) * 255
    new_img = new_img.astype('uint8')
    return new_img

def view_image(X_l, X_ab, model, img_size):
    """
    Input: L and ab chanels for test set and generator model.
    Output: Displays images.
    """
    img_lst_pred = model.predict(X_l)

    #Merge L and predicted AB
    img_lst_gen = [lab_to_rgb(X_l[i], img_lst_pred[i], img_size) for i in range(len(img_lst_pred))]
    img_lst_gen = [Image.fromarray(image,'RGB') for image in img_lst_gen]

    #Merge L and AB to produce original true images
    img_lst_real_arr = [lab_to_rgb(X_l[i], X_ab[i], img_size) for i in range(len(X_l))]
    img_lst_real = [Image.fromarray(image,'RGB') for image in img_lst_real_arr]

    img_lst_gray = [grayscale_image(image) for image in img_lst_real_arr]
    img_lst_gray = [Image.fromarray(i,'L') for i in img_lst_gray]

    for i in range(len(img_lst_gen)):
        img_lst_gen[i].show()
        img_lst_real[i].show()
        img_lst_gray[i].show()
    if not os.path.exists('../results/' + time.strftime('%d')):
        os.makedirs('../results/' + time.strftime('%d'))
    img_lst_gen[0].save('../results/' + time.strftime('%d') + '/' + time.strftime('%H:%M:%S') + '.png')
    return img_lst_pred

def predict_on_generated_images(images,model):
    """
    Input: Predicted Colorized images and Discriminator model.
    Output: Discriminator's prediction. 0=fake and 1=real.
    """
    real_or_fake = model.predict(images)
    return real_or_fake

def pres_title_slide():
    """
    Create title slide for presentation. Shows 2 9x8 grid of images. One is black and white images and the other is color images.
    """
    (X_train_l,X_train_ab) = load_images('../data/X_train.p')
    X_l = X_train_l[:72]
    X_ab = X_train_ab[:72]

    img_lst_arr = [lab_to_rgb(X_l[i], X_ab[i]) for i in range(len(X_l))]
    img_gray_lst = np.array([grayscale_image(image) for image in img_lst_arr])

    img_lst_pred = gen_model.predict(X_l)
    img_color_lst = np.array([lab_to_rgb(X_l[i], img_lst_pred[i]) for i in range(len(img_lst_pred))])

    gray_image = np.ones((9*256,8*256))
    k = 0
    for i in range(9):
        for j in range(8):
            gray_image[i*256:(i+1)*256,j*256:(j+1)*256] = img_gray_lst[k]
            k += 1
    gray_image = gray_image.astype('uint8')

    color_image = np.ones((9*256,8*256,3))
    k = 0
    for i in range(9):
        for j in range(8):
            color_image[i*256:(i+1)*256,j*256:(j+1)*256,:] = img_color_lst[k]
            k += 1
    color_image = color_image.astype('uint8')

    gray_image = Image.fromarray(gray_image,'L')
    gray_image.show()
    color_image = Image.fromarray(color_image,'RGB')
    color_image.show()

if __name__ =='__main__':
    gen_model = load_model('../models/Forest/gen_model_full_batch_10.h5')
    disc_model = load_model('../models/Forest/disc_model_full_batch_10.h5')
    # (X_test_l,X_test_ab) = load_images('../data/X_test.p')
    (X_test_l,X_test_ab) = load_images('../data/X_train.p')
    img_size = X_test_l.shape[1]
    rand_arr = np.arange(len(X_test_l))
    np.random.shuffle(rand_arr)
    img_results = view_image(X_test_l[rand_arr[22:24]], X_test_ab[rand_arr[22:24]], gen_model, img_size)

    results = predict_on_generated_images(img_results, disc_model)
    print(results)
