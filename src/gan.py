from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Activation, BatchNormalization, UpSampling2D, Dropout, Flatten, Dense, Input, LeakyReLU, Conv2DTranspose
from keras.optimizers import Adam
from keras.models import Sequential
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
        self.generator.compile(loss='categorical_crossentropy', optimizer=opt)
        print('Generator Summary...')
        print(self.generator.summary())

        self.discriminator = self.build_discriminator()
        opt = Adam(lr=.0001)
        self.discriminator.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        print('Discriminator Summary...')
        print(self.discriminator.summary())

        gan_input = Input(shape=self.g_input_shape)
        img_color = self.generator(gan_input)
        self.discriminator.trainable = False
        real_or_fake = self.discriminator(img_color)
        self.gan = Model(gan_input,real_or_fake)
        opt = Adam(lr=.001)
        self.gan.compile(loss='categorical_crossentropy', optimizer=opt)
        print('\n')
        print('GAN summary...')
        print(self.gan.summary())

    def build_generator(self):
        model = Sequential()
        model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=self.g_input_shape))
        model.add(Conv2D(64, (3, 3), padding='same', strides=2, activation='relu'))
        model.add(BatchNormalization())
        # model = MaxPooling2D(pool_size=(2, 2))(model)

        model.add(Conv2D(128, (3, 3), padding='same', activation='relu', strides=2))
        model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
        model.add(BatchNormalization())
        # model = MaxPooling2D(pool_size=(2, 2))(model)

        model.add(Conv2D(256, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(UpSampling2D(size=(2,2)))
        model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
        model.add(BatchNormalization())

        model.add(UpSampling2D(size=(2,2)))
        model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        model.add(BatchNormalization())

        model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(2, (3, 3), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('tanh'))
        # self.model = BatchNormalization()(self.model)
        # self.model = merge(inputs=[self.g_input, self.model], mode='concat')
        # self.model = Activation('linear')(self.model)
        return model

    def build_discriminator(self):
        model = Sequential()
        model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=self.d_input_shape))
        model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(.25))

        model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(.25))

        model.add(Flatten())
        model.add(Dense(512))
        model.add(LeakyReLU(.2))
        model.add(BatchNormalization())
        model.add(Dropout(.5))
        model.add(Dense(2))
        model.add(Activation('softmax'))

        return model

    def save_g(self,name):
        self.generator.save('../models/' + name + '.h5')

    def save_d(self,name):
        self.discriminator.save('../models/' + name + '.h5')

    def pre_train_discriminator(self, X_train_L, X_train_AB, X_test_L, X_test_AB):
        generated_images = self.generator.predict(X_train_L)
        X_train = np.concatenate((X_train_AB,generated_images))
        n = len(X_train_L)
        y_train = np.array([[0,1]] * n + [[1,0]] * n)
        rand_arr = np.arange(len(X_train))
        np.random.shuffle(rand_arr)
        X_train = X_train[rand_arr]
        y_train = y_train[rand_arr]

        test_generated_images = self.generator.predict(X_test_L)
        X_test = np.concatenate((X_test_AB,test_generated_images))
        n = len(X_test_L)
        y_test = np.array([[0,1]] * n + [[1,0]] * n)
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
            y_train_real = np.concatenate((np.zeros([n,1]), np.zeros([n,1])), axis=-1)
            y_train_fake = np.concatenate((np.ones([n,1]), np.zeros([n,1])), axis=-1)

            noise = np.random.rand(batch_size,32,32,2) * 2 -1

            d_loss = self.discriminator.fit(x=X_train_AB[:batch_size],y=y_train_real)
            d_loss = self.discriminator.fit(x=noise,y=y_train_fake)
            d_loss = self.discriminator.fit(x=generated_images,y=y_train_fake)
            d_losses.append(d_loss.history['loss'][-1])
            d_acc.append(d_loss.history['acc'][-1])
            print('d_loss:', d_loss.history['loss'][-1])
            # print("Discriminator Accuracy: ", disc_acc)

            #train GAN on grayscaled images , set output class to colorized
            n = batch_size
            y_train = np.concatenate((np.zeros([n,1]), np.ones([n,1])), axis=-1)
            np.random.shuffle(X_train)

            g_loss = self.gan.fit(x=X_train[:batch_size],y=y_train)

            g_losses.append(g_loss.history['loss'][-1])
            print('Generator Loss: ', g_loss.history['loss'][-1])
            disc_acc = d_loss.history['acc'][-1]
            if disc_acc < .9:
                self.pre_train_discriminator(X_train_L, X_train_AB, X_test_L, X_test_AB)
            if e % 5 == 4:
                print(e + 1,"batches done")
            if e % 100 == 24:
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

    batch_epochs = 1000
    batch_size = 64

    gan = GAN()
    gan.train(X_train_L, X_train_AB, X_test_L, X_test_AB, batch_epochs, batch_size)
