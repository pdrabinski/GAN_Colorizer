from keras.models import load_model
import pickle
import numpy as np
from PIL import Image
import time
import os
import gan
from skimage import color

def load_images(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def lab_to_rgb(l_layer, ab_layers):
    new_img = np.zeros((32,32,2))
    rescaled_l = np.zeros((32,32,1))
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

def view_image(X_l, X_ab, model):
    img_lst_pred = model.predict(X_l)

    #Merge L and predicted AB
    img_lst_gen = [lab_to_rgb(X_l[i], img_lst_pred[i]) for i in range(len(img_lst_pred))]
    img_lst_gen = [Image.fromarray(image,'RGB') for image in img_lst_gen]

    #Merge L and AB to produce original true images
    img_lst_real = [lab_to_rgb(X_l[i], X_ab[i]) for i in range(len(X_l))]
    img_lst_real = [Image.fromarray(image,'RGB') for image in img_lst_real]

    for i in range(len(img_lst_gen)):
        img_lst_gen[i].show()
        img_lst_real[i].show()
    if not os.path.exists('../test_images/' + time.strftime('%d')):
        os.makedirs('../test_images/' + time.strftime('%d'))
    img_lst_gen[0].save('../test_images/' + time.strftime('%d') + '/' + time.strftime('%H:%M:%S') + '.png')
    return img_lst_pred

def predict_on_generated_images(images,model):
    real_or_fake = model.predict(images)
    return real_or_fake

if __name__ =='__main__':
    gen_model = load_model('../models/gen_model_256_100.h5')
    disc_model = load_model('../models/disc_model_256_100.h5')
    (X_test_l,X_test_ab) = load_images('../data/X_test.p')
    rand_arr = np.arange(len(X_test_l))
    np.random.shuffle(rand_arr)
    img_results = view_image(X_test_l[rand_arr[:2]], X_test_ab[rand_arr[:2]], gen_model)
    results = predict_on_generated_images(img_results, disc_model)
    print(results)
