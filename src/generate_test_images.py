from keras.models import load_model
import pickle
import numpy as np
from PIL import Image
import gan

def load_images(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def view_image(images, model):
    print(type(model))
    img_lst = model.g.predict(images)
    print(img_lst)
    img_lst = [Image.fromarray(image,'RGB') for image in img_lst]
    for i in img_lst:
        i.show()

if __name__ =='__main__':
    model = load_model('../models/model_128_100.h5')
    images = load_images('../data/X_test.p')
    np.random.shuffle(images)
    view_image(images[:5], model)
