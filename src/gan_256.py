from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Activation, BatchNormalization, UpSampling2D, Dropout, Flatten, Dense, Input, LeakyReLU, Conv2DTranspose,AveragePooling2D, Concatenate
from keras.optimizers import Adam
from keras.models import Sequential
from tensorflow import set_random_seed
import numpy as np
import matplotlib.pyplot as plt
import pickle
import keras.backend as K

np.random.seed(1)
set_random_seed(1)

def load_images(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)

class GAN():
    def __init__(self):
        """
        Initialize the GAN. Includes compiling the generator and the discriminator separately and then together as the GAN.
        """
        self.g_input_shape = (256,256,1)
        self.d_input_shape = (256,256,2)

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
        """
        Returns generator as Keras model.
        """
        g_input = Input(shape=self.g_input_shape)
        #128 x 128
        conv1 = Conv2D(64, (3, 3), padding='same', strides=2)(g_input)
        conv1 = BatchNormalization()(conv1)
        conv1 = Activation('relu')(conv1)

        conv2 = Conv2D(128, (3, 3), padding='same', strides=1)(conv1)
        conv2 = BatchNormalization()(conv2)
        conv2 = Activation('relu')(conv2)

        #64 x 64
        conv3 = Conv2D(128, (3, 3), padding='same', strides=2)(conv2)
        conv3 = BatchNormalization()(conv3)
        conv3 = Activation('relu')(conv3)

        conv4 = Conv2D(256, (3, 3), padding='same', strides=1)(conv3)
        conv4 = BatchNormalization()(conv4)
        conv4 = Activation('relu')(conv4)

        #32 x 32
        conv5 = Conv2D(512, (3, 3), padding='same', strides=2)(conv4)
        conv5 = BatchNormalization()(conv5)
        conv5 = Activation('relu')(conv5)

        # conv6 = Conv2D(512, (3, 3), padding='same', strides=1)(conv5)
        # conv6 = BatchNormalization()(conv6)
        # conv6 = Activation('relu')(conv6)

        #64 x 64
        conv7 = UpSampling2D(size=(2, 2))(conv5)
        conv7 = Conv2D(256, (3, 3), padding='same')(conv7)
        conv7 = BatchNormalization()(conv7)
        conv7 = Activation('relu')(conv7)
        conv7 = Concatenate(axis=-1)([conv7,conv4])

        conv8 = Conv2D(256, (3, 3), padding='same')(conv7)
        conv8 = BatchNormalization()(conv8)
        conv8 = Activation('relu')(conv8)

        #128 x 128
        up2 = UpSampling2D(size=(2, 2))(conv8)
        conv9 = Conv2D(128, (3,3), padding='same')(up2)
        conv9 = BatchNormalization()(conv9)
        conv9 = Activation('relu')(conv9)
        conv9 = Concatenate(axis=-1)([conv9,conv2])

        conv10 = Conv2D(128, (3, 3), padding='same')(conv9)
        conv10 = BatchNormalization()(conv10)
        conv10 = Activation('relu')(conv10)

        up3 = UpSampling2D(size=(2, 2))(conv10)
        conv11 = Conv2D(64, (3,3), padding='same')(up3)
        conv11 = BatchNormalization()(conv11)
        conv11 = Activation('relu')(conv11)

        conv12 = Conv2D(2, (3, 3), padding='same')(conv11)
        # conv12 = BatchNormalization()(conv12)
        conv12 = Activation('tanh')(conv12)

        model = Model(inputs=g_input,outputs=conv12)
        return model

    def build_discriminator(self):
        """
        Returns discriminator as Keras model.
        """
        model = Sequential()
        model.add(Conv2D(32, (3, 3), padding='same', input_shape=self.d_input_shape, strides=2))
        model.add(LeakyReLU(.2))
        # model.add(Dropout(.25))

        model.add(AveragePooling2D(pool_size=(2, 2)))
        model.add(Conv2D(64, (3, 3), padding='same',strides=1))
        model.add(BatchNormalization())
        model.add(LeakyReLU(.2))
        model.add(Dropout(.25))

        model.add(AveragePooling2D(pool_size=(2, 2)))
        model.add(Conv2D(128, (3, 3), padding='same',strides=1))
        model.add(BatchNormalization())
        model.add(LeakyReLU(.2))
        model.add(Dropout(.25))

        # model.add(AveragePooling2D(pool_size=(2, 2)))
        model.add(Conv2D(256, (3, 3), padding='same',strides=2))
        model.add(BatchNormalization())
        model.add(LeakyReLU(.2))
        model.add(Dropout(.5))

        # model.add(Conv2D(512, (3, 3), padding='same',strides=2))
        # model.add(BatchNormalization())
        # model.add(LeakyReLU(.2))
        # model.add(Dropout(.25))

        model.add(Flatten())
        # model.add(Dense(512))
        # model.add(Dropout(.5))
        model.add(Dense(1))
        model.add(Activation('sigmoid'))

        return model

    def train_discriminator(self, X_train_L, X_train_AB, X_test_L, X_test_AB):
        """
        Function to train the discriminator. Called when discriminator accuracy falls below and a specified threshold.
        """
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
        if metrics[1] < .90:
            self.train_discriminator(X_train_L, X_train_AB, X_test_L, X_test_AB)

    def train(self, X_train_L, X_train_AB, X_test_L, X_test_AB, epochs):
        """
        Training loop for GAN. First the discriminator is fit with real and fake images. Next the Generator is fit. This is possible because the weights in the Discriminator are fixed and not affected by back propagation.
        Inputs: X_train L channel, X_train AB channels, X_test L channel, X_test AB channels, number of epochs.
        Outputs: Models are saved and loss/acc plots saved.
        """

        # self.train_discriminator(X_train_L, X_train_AB, X_test_L, X_test_AB)
        g_losses = []
        d_losses = []
        d_acc = []
        X_train = X_train_L
        n = len(X_train)
        y_train_fake = np.zeros([n,1])
        y_train_real = np.ones([n,1])
        for e in range(epochs):
            #generate images
            np.random.shuffle(X_train)
            generated_images = self.generator.predict(X_train, verbose=1)
            np.random.shuffle(X_train_AB)

            #Train Discriminator
            d_loss = self.discriminator.fit(x=X_train_AB,y=y_train_real,batch_size=16,epochs=1)
            if e % 3 == 2:
                noise = np.random.rand(n,256,256,2) * 2 -1
                d_loss = self.discriminator.fit(x=noise,y=y_train_fake, batch_size=16, epochs=1)
            d_loss = self.discriminator.fit(x=generated_images,y=y_train_fake,batch_size=16,epochs=1)
            d_losses.append(d_loss.history['loss'][-1])
            d_acc.append(d_loss.history['acc'][-1])
            print('d_loss:', d_loss.history['loss'][-1])
            # print("Discriminator Accuracy: ", disc_acc)

            #train GAN on grayscaled images , set output class to colorized
            g_loss = self.gan.fit(x=X_train,y=y_train_real,batch_size=16,epochs=1)

            #Record Losses/Acc
            g_losses.append(g_loss.history['loss'][-1])
            print('Generator Loss: ', g_loss.history['loss'][-1])
            disc_acc = d_loss.history['acc'][-1]

            # Retrain Discriminator if accuracy drops below .8
            if disc_acc < .8 and e < (epochs / 2):
                self.train_discriminator(X_train_L, X_train_AB, X_test_L, X_test_AB)
            if e % 5 == 4:
                print(e + 1,"batches done")

        self.plot_losses(g_losses,'Generative Loss', epochs)
        self.plot_losses(d_acc, 'Discriminative Accuracy',epochs)
        self.generator.save('../models/gen_model_full_batch_' + str(epochs)+'.h5')
        self.discriminator.save('../models/disc_model_full_batch_' + str(epochs)+'.h5')

    def plot_losses(self, metric, label, epochs):
        """
        Plot the loss/acc of the generator/discriminator.
        Inputs: metric, label of graph, number of epochs (for file name)
        """
        plt.plot(metric, label=label)
        plt.title('GAN Accuracy and Loss Over ' + str(epochs) + ' Epochs')
        plt.savefig('../plots/plot_' + str(epochs) + '_epochs.png')
        # plt.close()

if __name__ == '__main__':
    (X_train_L, X_train_AB) = load_images('../data/X_train.p')
    X_train_L = X_train_L.astype('float32')
    X_train_AB = X_train_AB.astype('float32')
    print('X_train done...')
    (X_test_L, X_test_AB) = load_images('../data/X_test.p')
    X_test_L = X_test_L.astype('float32')
    X_test_AB = X_test_AB.astype('float32')
    print('X_test done...')

    epochs = 30

    gan = GAN()
    gan.train(X_train_L, X_train_AB, X_test_L, X_test_AB, epochs)
