import pickle
import numpy as np
from gan import Generator, Discriminator, GAN

def load_images(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def train_discriminator(X_train, X_train_true, X_test, X_test_true, g ,d):
    generated_images = g.predict(X_train)
    X_train = np.concatenate((X_train_true,generated_images))
    n = len(X_train)
    y_train = np.zeros([n,2])
    y_train[:n,1] = 1
    y_train[n:,0] = 1

    test_generated_images = g.predict(X_test)
    X_test = np.concatenate((X_test_true,test_generated_images))
    n = len(X_test)
    y_test = np.zeros([n,2])
    y_test[:n,1] = 1
    y_test[n:,0] = 1

    d.make_trainable(True)
    d.fit(X_train,y_train,X_test,y_test,epochs=1)
    # y_pred = d.predict(X_train)
    # discriminator_accuracy(y_pred,y_train)

# def discriminator_accuracy(y_pred,y_true):
#     y_pred = np.argmax(y_pred, axis=1)
#     y_true = np.argmax(y_true, axis=1)
#     n_wrong = np.sum(np.absolute(y_pred - y_true))
#     n = len(y_pred)
#     n_right = n - n_wrong
#     accuracy = round((n_right)/n * 100,2)
#     print('Accuracy:',accuracy)
#     print(n_right,"of",n,"correct \n")

if __name__ == '__main__':
    #Load images
    X_train = load_images('../data/X_train.p')
    print('X_train done...')
    X_test = load_images('../data/X_test.p')
    print('X_test done...')
    X_train_true = load_images('../data/X_train_true.p')
    print('X_train_true done...')
    X_test_true = load_images('../data/X_test_true.p')
    print('X_test_true done...')

    # Create Stacked GAN
    bw_shape = X_train.shape[1:]
    color_shape = X_train_true.shape[1:]

    g = Generator()
    g_tensor = g.build(input_shape=bw_shape)
    d = Discriminator()
    d_tensor = d.build(input_shape=color_shape)
    generator = g.compile()
    discriminator = d.compile()

    gan = GAN()
    gan.compile(g=generator,d=discriminator,input_shape=bw_shape)

    #Train the Discriminator
    train_discriminator(X_train, X_train_true, X_test, X_test_true, g,d)
