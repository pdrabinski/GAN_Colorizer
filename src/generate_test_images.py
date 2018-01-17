from keras.models import load_model
import pickle
import numpy as np
from PIL import Image
import time
import os
import gan

def load_images(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def view_image(images, model):
    img_lst = model.predict(images)
    img_lst = [img * 256 for img in img_lst]
    print(img_lst)
    print(img_lst[1].shape)
    img_lst = [Image.fromarray(image,'RGB') for image in img_lst]
    for i in img_lst:
        i.show()
    if not os.path.exists('../images/test_images/' + time.strftime('%d')):
        os.makedirs('../images/test_images/' + time.strftime('%d'))
    img_lst[0].save('../images/test_images/' + time.strftime('%d') + '/' + time.strftime('%H:%M:%S') + '.png')

if __name__ =='__main__':
    model = load_model('../models/model_512_20.h5')
    images = load_images('../data/X_test.p')
    np.random.shuffle(images)
    view_image(images[:5], model)
