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

def lab_to_rgb(image):
    rgb = color.lab2rgb(image)
    return rgb

def view_image(images, model):
    img_lst = model.predict(images)
    # img_lst = [(img + 1) * 128 for img in img_lst]
    # print(img_lst)
    print(img_lst[1].shape)
    img_lst = [Image.fromarray(lab_to_rgb(image),'RGB') for image in img_lst]
    for i in img_lst:
        i.show()
    if not os.path.exists('../test_images/' + time.strftime('%d')):
        os.makedirs('../test_images/' + time.strftime('%d'))
    img_lst[0].save('../test_images/' + time.strftime('%d') + '/' + time.strftime('%H:%M:%S') + '.png')

def predict_on_generated_images(images,model):
    real_or_fake = model.predict(images)
    return real_or_fake

if __name__ =='__main__':
    model = load_model('../models/gen_model_512_50.h5')
    images = load_images('../data/X_test.p')
    np.random.shuffle(images)
    view_image(images[:5], model)
