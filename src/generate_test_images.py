from keras.models import load_model
import pickle
import numpy as np
from PIL import Image
import gan

def load_images(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def view_image(image, model):
    model.predict(image)
    img = Image.fromarray(image,'RGB')
    img.show()
    pass

if __name__ =='__main__':
    model = load_model('../models/model_64_100.h5')
    images = load_images('../data/X_test.p')
    np.random.shuffle(images)
    view_image(images[0], model)
