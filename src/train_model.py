import pickle
import numpy as np
from gan import Generator, Discriminator, GAN
import matplotlib.pyplot as plt

def load_images(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def train_discriminator(X_train, X_train_true, X_test, X_test_true, gan):
    generated_images = gan.g.predict(X_train)
    X_train = np.concatenate((X_train_true,generated_images))
    n = len(X_train)
    y_train = np.zeros([n,2])
    y_train[:n,1] = 1
    y_train[n:,0] = 1

    test_generated_images = gan.predict(X_test)
    X_test = np.concatenate((X_test_true,test_generated_images))
    n = len(X_test)
    y_test = np.zeros([n,2])
    y_test[:n,1] = 1
    y_test[n:,0] = 1

    gan.d.make_trainable(True)
    gan.d.fit(X_train,y_train,X_test,y_test,epochs=1)
    y_pred = gan.d.predict(X_train)
    discriminator_accuracy(y_pred,y_train)

def discriminator_accuracy(y_pred,y_true):
    y_pred = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_true, axis=1)
    n_wrong = np.sum(np.absolute(y_pred - y_true))
    n = len(y_pred)
    n_right = n - n_wrong
    accuracy = round((n_right)/n * 100,2)
    print('Accuracy:',accuracy)
    print(n_right,"of",n,"correct \n")

def train(X_train, X_test, X_train_true, X_test_true, batch_epochs, batch_size, gan):
    g_losses = []
    d_losses = []
    for e in range(batch_epochs):
        #generate images
        X_train_disc = X_train
        np.random.shuffle(X_train_disc)
        X_train_disc = X_train_disc[:batch_size]
        generated_images = gan.g.predict(X_train_disc)
        np.random.shuffle(X_train_true)
        #try shuffling generated images and true the same way
        X_train_disc = np.concatenate((X_train_true[:batch_size],generated_images))
        n = batch_size * 2
        y_train = np.zeros([n,2])
        y_train[:n,1] = 1
        y_train[n:,0] = 1

        #train discriminator
        gan.d.make_trainable(True)
        gan.d.compile()
        loss = gan.d.train_on_batch(X_train_disc,y_train)
        d_losses.append(loss)

        #train GAN on grayscaled images , set output class to colorized
        n = batch_size
        y_train = np.zeros([n,2])
        y_train[:,1] = 1
        gan.d.make_trainable(False)
        gan.d.compile()
        X_train_gen = X_train
        np.random.shuffle(X_train_gen)
        g_loss = gan.train_on_batch(X_train_gen[:batch_size],y_train)
        g_losses.append(g_loss)
        print(e,"batches done")

    plot_losses(g_losses,'Generative_Losses',batch_epochs, batch_size)
    plot_losses(d_losses,'Discriminative_Losses',batch_epochs, batch_size)
    gan.save('model_' + str(batch_size) + '_' + str(batch_epochs))

def plot_losses(losses,label, batch_epochs, batch_size):
    plt.plot(losses)
    plt.title(label)
    plt.savefig('../images/' + label + '_' + str(batch_size) + '_' + str(batch_epochs) + '.png')


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
    gan = GAN()
    gan.compile(input_shape=bw_shape, output_shape=color_shape)

    # Pre-train the Discriminator
    train_discriminator(X_train, X_train_true, X_test, X_test_true, gan)

    #Train GAN
    batch_size=512
    batch_epochs=20
    train(X_train, X_test, X_train_true, X_test_true, batch_epochs, batch_size, gan)
