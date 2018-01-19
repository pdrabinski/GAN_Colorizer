from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Activation, BatchNormalization, UpSampling2D, merge, Dropout, Flatten, Dense, Input, LeakyReLU
from keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
import pickle

def load_images(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)

class GAN():
    def __init__(self):
        self.g_input_shape = (32,32,1)
        self.d_input_shape = (32,32,2)

        self.generator = self.build_generator()
        opt = Adam(lr=.001)
        self.generator.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
        print('Generator Summary...')
        print(self.generator.summary())

        self.discriminator = self.build_discriminator()
        opt = Adam(lr=.0001)
        self.discriminator.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
        print('Discriminator Summary...')
        print(self.discriminator.summary())

        gan_input = Input(shape=self.g_input_shape)
        img_color = self.generator(gan_input)
        self.discriminator.trainable = False
        real_or_fake = self.discriminator(img_color)
        self.gan = Model(gan_input,real_or_fake)
        opt = Adam(lr=.001)
        self.gan.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
        print('\n')
        print('GAN summary...')
        print(self.gan.summary())

    def build_generator(self):
        g_input = Input(shape=self.g_input_shape)
        model = Conv2D(32, (3, 3), padding='same')(g_input)
        model = Activation('relu')(model)
        model = Conv2D(32, (3, 3), padding='same')(model)
        model = Activation('relu')(model)
        model = BatchNormalization()(model)
        model = MaxPooling2D(pool_size=(2, 2))(model)

        model = Conv2D(64, (3, 3), padding='same')(model)
        model = Activation('relu')(model)
        model = Conv2D(64, (3, 3), padding='same')(model)
        model = Activation('relu')(model)
        model = BatchNormalization()(model)
        model = MaxPooling2D(pool_size=(2, 2))(model)

        model = Conv2D(128, (3, 3), padding='same')(model)
        model = Activation('relu')(model)
        model = BatchNormalization()(model)

        model = UpSampling2D(size=(2,2))(model)
        model = Conv2D(128, (3, 3), padding='same')(model)
        model = Activation('relu')(model)
        model = BatchNormalization()(model)

        model = UpSampling2D(size=(2,2))(model)
        model = Conv2D(64, (3, 3), padding='same')(model)
        model = Activation('relu')(model)
        model = BatchNormalization()(model)

        model = Conv2D(32, (3, 3), padding='same')(model)
        model = Activation('relu')(model)
        model = Conv2D(2, (3, 3), padding='same')(model)
        model = Activation('sigmoid')(model)
        # self.model = BatchNormalization()(self.model)
        # self.model = merge(inputs=[self.g_input, self.model], mode='concat')
        # self.model = Activation('linear')(self.model)
        gen = Model(g_input, model)
        return gen

    def build_discriminator(self):
        d_input = Input(shape=self.d_input_shape)

        model = Conv2D(32, (3, 3), padding='same', activation='relu')(d_input)
        model = Conv2D(32, (3, 3), padding='same', activation='relu')(model)
        model = MaxPooling2D(pool_size=(2, 2))(model)
        model = Dropout(.25)(model)

        model = Conv2D(64, (3, 3), padding='same', activation='relu')(model)
        model = Conv2D(64, (3, 3), padding='same', activation='relu')(model)
        model = MaxPooling2D(pool_size=(2, 2))(model)
        model = Dropout(.25)(model)

        model = Flatten()(model)
        model = Dense(512)(model)
        model = LeakyReLU(.2)(model)
        model = Dropout(.5)(model)
        model = Dense(1)(model)
        model = Activation('sigmoid')(model)

        disc = Model(d_input,model)
        return disc

    def save_g(self,name):
        self.generator.save('../models/' + name + '.h5')

    def save_d(self,name):
        self.discriminator.save('../models/' + name + '.h5')

    def pre_train_discriminator(self, X_train_L, X_train_AB, X_test_L, X_test_AB):
        generated_images = self.generator.predict(X_train_L)
        X_train = np.concatenate((X_train_AB,generated_images))
        n = len(X_train_L)
        y_train = np.array([[1]] * n + [[0]] * n)
        rand_arr = np.arange(len(X_train))
        np.random.shuffle(rand_arr)
        X_train = X_train[rand_arr]
        y_train = y_train[rand_arr]

        test_generated_images = self.generator.predict(X_test_L)
        X_test = np.concatenate((X_test_AB,test_generated_images))
        n = len(X_test_L)
        y_test = np.array([[1]] * n + [[0]] * n)
        rand_arr = np.arange(len(X_test))
        np.random.shuffle(rand_arr)
        X_test = X_test[rand_arr]
        y_test = y_test[rand_arr]

        self.discriminator.fit(x=X_train,y=y_train,validation_data=(X_test,y_test),epochs=1)
        metrics = self.discriminator.evaluate(x=X_test, y=y_test)
        print('\n accuracy:',metrics[1])
        if metrics[1] < .95:
            self.pre_train_discriminator(X_train_L, X_train_AB, X_test_L, X_test_AB)

    def train(self, X_train_L, X_train_AB, X_test_L, X_test_AB, batch_epochs, batch_size):
        self.pre_train_discriminator(X_train_L, X_train_AB, X_test_L, X_test_AB)
        g_losses = []
        d_losses = []
        disc_acc = 0
        gen_acc = 0
        for e in range(batch_epochs):
            #generate images
            X_train_disc = X_train_L
            np.random.shuffle(X_train_disc)
            X_train_disc = X_train_disc[:batch_size]
            generated_images = self.generator.predict(X_train_disc)
            np.random.shuffle(X_train_AB)

            n = batch_size
            y_train_real = np.ones([n,1])
            y_train_fake = np.zeros([n,1])

            d_loss = self.discriminator.train_on_batch(X_train_AB[:batch_size],y_train_real)
            d_loss = self.discriminator.train_on_batch(generated_images,y_train_fake)
            d_losses.append(d_loss)
            disc_acc = d_loss[1]
            print("Discriminator Accuracy: ", disc_acc)

            #train GAN on grayscaled images , set output class to colorized
            n = batch_size
            y_train = np.ones([n])
            X_train_gen = X_train_L
            np.random.shuffle(X_train_gen)
            g_loss = self.gan.train_on_batch(X_train_gen[:batch_size],y_train)

            g_losses.append(g_loss)
            gen_acc = g_loss[1]
            print('Generator Accuracy: ', gen_acc)
            if disc_acc < .9:
                self.pre_train_discriminator(X_train_L, X_train_AB, X_test_L, X_test_AB)
            if e % 5 == 4:
                print(e + 1,"batches done")
            if e % 25 == 24:
                self.plot_losses(g_losses,'Generative_Losses',e, batch_size)
                self.plot_losses(d_losses,'Discriminative_Losses',e, batch_size)

        self.generator.save('gen_model_' + str(batch_size) + '_' + str(batch_epochs))
        self.discriminator.save('disc_model_' + str(batch_size) + '_' + str(batch_epochs))

    def plot_losses(self, losses, label, batch_epochs, batch_size):
        plt.plot(losses)
        plt.title(label)
        plt.savefig('../images/' + label + '_' + str(batch_size) + '_' + str(batch_epochs) + '_epochs.png')
        plt.close()

if __name__ == '__main__':
    (X_train_L, X_train_AB) = load_images('../data/X_train.p')
    print('X_train done...')
    (X_test_L, X_test_AB) = load_images('../data/X_test.p')
    print('X_test done...')

    batch_epochs = 100
    batch_size = 256

    gan = GAN()
    gan.train(X_train_L, X_train_AB, X_test_L, X_test_AB, batch_epochs, batch_size)
