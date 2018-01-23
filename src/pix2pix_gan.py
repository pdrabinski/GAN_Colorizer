from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Activation, BatchNormalization, UpSampling2D, Dropout, Flatten, Dense, Input, LeakyReLU, Conv2DTranspose,AveragePooling2D, Concatenate
from keras.optimizers import Adam
from keras.models import Sequential
from tensorflow import set_random_seed
import numpy as np
import matplotlib.pyplot as plt
import pickle

np.random.seed(1)
set_random_seed(1)

def load_images(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)

class GAN():
    def __init__(self):
        self.g_input_shape = (32,32,1)
        self.d_input_shape = (32,32,2)

        self.generator = self.build_generator()
        opt = Adam(lr=.001)
        self.generator.compile(loss='binary_crossentropy', optimizer=opt)
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
        self.gan.compile(loss='binary_crossentropy', optimizer=opt)
        print('\n')
        print('GAN summary...')
        print(self.gan.summary())

    def build_generator(self):
        g_input = Input(shape=self.g_input_shape)
        conv1 = Conv2D(32, (3, 3), padding='same')(g_input)
        conv1 = LeakyReLU(.2)(conv1)
        conv1 = BatchNormalization()(conv1)

        conv2 = Conv2D(64, (3, 3), padding='same', strides=2)(conv1)
        conv2 = LeakyReLU(.2)(conv2)
        conv2 = BatchNormalization()(conv2)

        conv3 = Conv2D(128, (3, 3), padding='same', strides=2)(conv2)
        conv3 = LeakyReLU(.2)(conv3)
        conv3 = BatchNormalization()(conv3)

        up_conv1 = Activation('relu')(conv3)
        up_conv1 = UpSampling2D(size=(2, 2))(up_conv1)
        up_conv1 = Conv2D(64, (3,3), padding='same')(up_conv1)
        up_conv1 = BatchNormalization()(up_conv1)
        up_conv1 = Dropout(.25)(up_conv1)
        up_conv1_unet = Concatenate(axis=-1)([up_conv1,conv2])

        up_conv2 = Activation('relu')(up_conv1_unet)
        up_conv2 = UpSampling2D(size=(2, 2))(up_conv2)
        up_conv2 = Conv2D(32, (3,3), padding='same')(up_conv2)
        up_conv2 = BatchNormalization()(up_conv2)
        up_conv2_unet = Concatenate(axis=-1)([up_conv2,conv1])

        up_conv3 = Activation('relu')(up_conv2_unet)
        up_conv3 = Conv2D(2,(3,3), padding='same')(up_conv3)
        up_conv3 = Activation('tanh')(up_conv3)

        model = Model(inputs=g_input,outputs=up_conv3)
        return model

    def build_discriminator(self):
        model = Sequential()
        model.add(Conv2D(32, (3, 3), padding='same', input_shape=self.d_input_shape, strides=2))
        # model.add(Conv2D(32, (3, 3), padding='same', activation='relu',strides=2))
        # model.add(AveragePooling2D(pool_size=(2, 2)))
        model.add(BatchNormalization())
        model.add(LeakyReLU(.2))
        # model.add(Dropout(.25))

        model.add(Conv2D(64, (3, 3), padding='same',strides=2))
        model.add(BatchNormalization())
        model.add(LeakyReLU(.2))
        # model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        # model.add(AveragePooling2D(pool_size=(2, 2)))
        # model.add(Dropout(.25))

        model.add(Flatten())
        # model.add(Dense(512))
        # model.add(LeakyReLU(.2))
        # model.add(BatchNormalization())
        # model.add(Dropout(.5))
        model.add(Dense(1))
        model.add(Activation('sigmoid'))

        #using discriminator from tensorflow classification
        # model = Sequential()
        # model.add(Conv2D(32, (5, 5), padding='same', activation='relu', input_shape=self.d_input_shape))
        # model.add(AveragePooling2D(pool_size=(2, 2)))
        # # model.add(Dropout(.25))
        #
        # model.add(Conv2D(64, (5, 5), padding='same', activation='relu'))
        # model.add(AveragePooling2D(pool_size=(2, 2)))
        #
        # model.add(Flatten())
        # model.add(Dense(1024))
        # model.add(LeakyReLU(.2))
        # model.add(BatchNormalization())
        # model.add(Dense(1))
        # model.add(Activation('sigmoid'))

        return model

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

        self.discriminator.fit(x=X_train,y=y_train,epochs=1)
        metrics = self.discriminator.evaluate(x=X_test, y=y_test)
        print('\n accuracy:',metrics[1])
        if metrics[1] < .95:
            self.pre_train_discriminator(X_train_L, X_train_AB, X_test_L, X_test_AB)

    def train(self, X_train_L, X_train_AB, X_test_L, X_test_AB, batch_epochs, batch_size):
        self.pre_train_discriminator(X_train_L, X_train_AB, X_test_L, X_test_AB)
        g_losses = []
        d_losses = []
        d_acc = []
        X_train = X_train_L
        for e in range(batch_epochs):
            #generate images
            np.random.shuffle(X_train)
            X_train_disc = X_train[:batch_size]
            generated_images = self.generator.predict(X_train_disc, verbose=1)
            np.random.shuffle(X_train_AB)

            n = batch_size
            y_train_fake = np.zeros([n,1])
            y_train_real = np.ones([n,1])

            d_loss = self.discriminator.fit(x=X_train_AB[:batch_size],y=y_train_real,epochs=1)
            if e % 15 == 14:
                noise = np.random.rand(batch_size,32,32,2) * 2 -1
                d_loss = self.discriminator.fit(x=noise,y=y_train_fake, epochs=1)
            d_loss = self.discriminator.fit(x=generated_images,y=y_train_fake, epochs=1)
            d_losses.append(d_loss.history['loss'][-1])
            d_acc.append(d_loss.history['acc'][-1])
            print('d_loss:', d_loss.history['loss'][-1])
            # print("Discriminator Accuracy: ", disc_acc)

            #train GAN on grayscaled images , set output class to colorized
            n = batch_size
            y_train = np.ones([n,1])
            # y_train = np.concatenate((np.zeros([n,1]), np.ones([n,1])), axis=-1)
            np.random.shuffle(X_train)

            g_loss = self.gan.fit(x=X_train[:batch_size],y=y_train,epochs=1)

            g_losses.append(g_loss.history['loss'][-1])
            print('Generator Loss: ', g_loss.history['loss'][-1])
            disc_acc = d_loss.history['acc'][-1]
            if disc_acc < .8:
                self.pre_train_discriminator(X_train_L, X_train_AB, X_test_L, X_test_AB)
            if e % 5 == 4:
                print(e + 1,"batches done")
            if e % 100 == 99:
                self.plot_losses(g_losses,'Generative_Losses',e, batch_size)
                self.plot_losses(d_acc,'Discriminative_Losses',e, batch_size)

        self.generator.save('../models/gen_model_' + str(batch_size) + '_' + str(batch_epochs)+'.h5')
        self.discriminator.save('../models/disc_model_' + str(batch_size) + '_' + str(batch_epochs)+'.h5')

    def plot_losses(self, losses, label, batch_epochs, batch_size):
        plt.plot(losses)
        plt.title(label)
        plt.savefig('../images/' + label + '_' + str(batch_size) + '_' + str(batch_epochs) + '_epochs.png')
        plt.close()

if __name__ == '__main__':
    (X_train_L, X_train_AB) = load_images('../data/X_train.p')
    X_train_L = X_train_L.astype('float32')
    X_train_AB = X_train_AB.astype('float32')
    print('X_train done...')
    (X_test_L, X_test_AB) = load_images('../data/X_test.p')
    X_test_L = X_test_L.astype('float32')
    X_test_AB = X_test_AB.astype('float32')
    print('X_test done...')

    batch_epochs = 200
    batch_size = 128

    gan = GAN()
    gan.train(X_train_L, X_train_AB, X_test_L, X_test_AB, batch_epochs, batch_size)
