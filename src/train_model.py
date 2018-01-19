import pickle
import numpy as np
from gan import GAN
import matplotlib.pyplot as plt

def load_images(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def train_discriminator_whole_batch(X_train_L, X_train_AB, X_test_L, X_test_AB, gan):
    generated_images = gan.g.predict(X_train_L)
    X_train_concat = np.concatenate((X_train_AB,generated_images))
    n = len(X_train_L)
    y_train = np.zeros([2 * n,1])
    y_train[:n] = 1
    y_train[n:] = 0
    rand_arr = np.arange(len(X_train_concat))
    np.random.shuffle(rand_arr)
    X_train_concat = X_train_concat[rand_arr]
    y_train = y_train[rand_arr]
    # print(y_train)

    test_generated_images = gan.predict(X_test_L)
    X_test_concat = np.concatenate((X_test_AB,test_generated_images))
    n = len(X_test_L)
    y_test_concat = np.zeros([2 * n,1])
    y_test_concat[:n] = 1
    y_test_concat[n:] = 0
    rand_arr = np.arange(len(X_test_concat))
    np.random.shuffle(rand_arr)
    X_test_concat = X_test_concat[rand_arr]
    y_test_concat = y_test_concat[rand_arr]
    # print(y_test_concat)

    gan.d.make_trainable(True)
    gan.d.compile()
    gan.d.fit(X_train_concat,y_train,X_test_concat,y_test_concat,epochs=1)
    metrics = gan.d.evaluate(x=X_test_concat, y=y_test_concat)
    print('accuracy:',metrics[1])
    if metrics[1] < .95:
        train_discriminator_whole_batch(X_train_L, X_train_AB, X_test_L, X_test_AB, gan)

    # y_pred = gan.d.predict(X_test_concat)
    # print(y_pred)
    # discriminator_accuracy(y_pred,y_test_concat)

def discriminator_accuracy(y_pred,y_true):
    n = len(y_pred)
    n_right = np.sum(y_pred == y_true)
    accuracy = round((n_right)/n * 100,2)
    print('Accuracy:',accuracy)
    print(n_right,"of",n,"correct \n")

def train(X_train_L, X_train_AB, X_test_L, X_test_AB, batch_epochs, batch_size, gan):
    g_losses = []
    d_losses = []
    disc_acc = 0
    gen_acc = 0
    for e in range(batch_epochs):
        #generate images
        # print('training discriminator...')
        X_train_disc = X_train_L
        np.random.shuffle(X_train_disc)
        X_train_disc = X_train_disc[:batch_size]
        generated_images = gan.g.predict(X_train_disc)
        np.random.shuffle(X_train_AB)
        X_train_disc = np.concatenate((X_train_AB[:batch_size],generated_images))
        n = batch_size
        y_train = np.zeros([n * 2,1])
        y_train[:n] = 1
        rand_arr = np.arange(len(X_train_disc))
        np.random.shuffle(rand_arr)
        X_train_disc = X_train_disc[rand_arr]
        y_train = y_train[rand_arr]

        #train discriminator
        gan.d.make_trainable(True)
        gan.d.compile()
        d_loss = gan.d.train_on_batch(X_train_disc,y_train)
        d_losses.append(d_loss)
        disc_acc = d_loss[1]
        print("Discriminator Accuracy: ", disc_acc)

        #train GAN on grayscaled images , set output class to colorized
        # print('training generator...')
        n = batch_size
        y_train = np.ones([n])
        gan.d.make_trainable(False)
        gan.d.compile()
        X_train_gen = X_train_L
        np.random.shuffle(X_train_gen)
        g_loss = gan.train_on_batch(X_train_gen[:batch_size],y_train)
        g_losses.append(g_loss)
        gen_acc = g_loss[1]
        g_losses.append(gen_acc)
        print('Generator Accuracy: ', gen_acc)
        if e % 5 == 4:
            print(e + 1,"batches done")
        if e % 25 == 24:
            plot_losses(g_losses,'Generative_Losses',e, batch_size)
            plot_losses(d_losses,'Discriminative_Losses',e, batch_size)

    # plot_losses(g_losses,'Generative_Losses',batch_epochs, batch_size)
    # plot_losses(d_losses,'Discriminative_Losses',batch_epochs, batch_size)
    gan.save_g('gen_model_' + str(batch_size) + '_' + str(batch_epochs))
    gan.save_d('disc_model_' + str(batch_size) + '_' + str(batch_epochs))

def plot_losses(losses,label, batch_epochs, batch_size):
    plt.plot(losses)
    plt.title(label)
    plt.savefig('../images/' + label + '_' + str(batch_size) + '_' + str(batch_epochs) + '_epochs.png')
    plt.close()


if __name__ == '__main__':
    #Load images
    (X_train_L, X_train_AB) = load_images('../data/X_train.p')
    print('X_train done...')
    (X_test_L, X_test_AB) = load_images('../data/X_test.p')
    print('X_test done...')
    # X_train_true = load_images('../data/X_train_true.p')
    # print('X_train_true done...')
    # X_test_true = load_images('../data/X_test_true.p')
    # print('X_test_true done...')

    # Create Stacked GAN
    bw_shape = X_train_L.shape[1:]
    color_shape = X_train_AB.shape[1:]

    gan = GAN()
    gan.compile(input_shape=bw_shape, output_shape=color_shape)

    # Pre-train the Discriminator
    train_discriminator_whole_batch(X_train_L, X_train_AB, X_test_L, X_test_AB, gan)

    #Train GAN
    batch_size=512
    batch_epochs=50
    train(X_train_L, X_train_AB, X_test_L, X_test_AB, batch_epochs, batch_size, gan)
