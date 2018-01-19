from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Activation, BatchNormalization, UpSampling2D, merge, Dropout, Flatten, Dense, Input, LeakyReLU
from keras.optimizers import Adam

class Generator():
    def build(self,input_shape):
        # bw_image = Sequential()
        # bw_image.add(Dense(shape=input_shape))

        self.g_input = Input(shape=input_shape)
        self.model = Conv2D(32, (3, 3), padding='same')(self.g_input)
        self.model = Activation('relu')(self.model)
        self.model = Conv2D(32, (3, 3), padding='same')(self.model)
        self.model = Activation('relu')(self.model)
        self.model = BatchNormalization()(self.model)
        self.model = MaxPooling2D(pool_size=(2, 2))(self.model)

        self.model = Conv2D(64, (3, 3), padding='same')(self.model)
        self.model = Activation('relu')(self.model)
        self.model = Conv2D(64, (3, 3), padding='same')(self.model)
        self.model = Activation('relu')(self.model)
        self.model = BatchNormalization()(self.model)
        self.model = MaxPooling2D(pool_size=(2, 2))(self.model)

        self.model = Conv2D(128, (3, 3), padding='same')(self.model)
        self.model = Activation('relu')(self.model)
        self.model = BatchNormalization()(self.model)

        self.model = UpSampling2D(size=(2,2))(self.model)
        self.model = Conv2D(128, (3, 3), padding='same')(self.model)
        self.model = Activation('relu')(self.model)
        self.model = BatchNormalization()(self.model)

        self.model = UpSampling2D(size=(2,2))(self.model)
        self.model = Conv2D(64, (3, 3), padding='same')(self.model)
        self.model = Activation('relu')(self.model)
        self.model = BatchNormalization()(self.model)

        self.model = Conv2D(32, (3, 3), padding='same')(self.model)
        self.model = Activation('relu')(self.model)
        self.model = Conv2D(2, (3, 3), padding='same')(self.model)
        self.model = Activation('sigmoid')(self.model)
        # self.model = merge(inputs=[self.g_input, self.model], mode='concat')
        # self.model = Activation('linear')(self.model)

    def compile(self):
        self.generator = Model(self.g_input, self.model)
        opt = Adam(lr=.001)
        self.generator.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
        print('\n')
        print('Generator summary...\n')
        print(self.generator.summary())
        return self.generator

    def freeze_weights(self,val):
        self.generator.trainable = val
        for layer in self.generator.layers:
            layer.trainable = val

    def predict(self, X):
        #need to fix steps
        return self.generator.predict(X, batch_size=32, verbose=1)

    def fit(self, X_train, X_test, y_train, y_test, batch_size=32, epochs=100):
        self.generator.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(X_test,y_test), shuffle=True)

    def save(self,name):
        self.generator.save('../models/' + name)


class Discriminator():
    def build(self, input_shape):
        self.d_input = Input(shape=input_shape)

        self.model = Conv2D(32, (3, 3), padding='same', activation='relu')(self.d_input)
        self.model = Conv2D(32, (3, 3), padding='same', activation='relu', strides=2)(self.d_input)
        self.model = Dropout(.25)(self.model)

        self.model = Conv2D(64, (3, 3), padding='same', activation='relu')(self.model)
        self.model = Conv2D(64, (3, 3), padding='same', activation='relu', strides=2)(self.d_input)
        self.model = Dropout(.25)(self.model)

        self.model = Flatten()(self.model)
        self.model = Dense(512)(self.model)
        self.model = LeakyReLU(.2)(self.model)
        self.model = Dropout(.5)(self.model)
        self.model = Dense(1)(self.model)
        self.model = Activation('sigmoid')(self.model)


        # self.model = LeakyReLU(.2)(self.model)
        # self.model = Dropout(.25)(self.model)
        #
        # self.model = Conv2D(32, (3, 3), padding='same', strides=2)(self.d_input)
        # self.model = LeakyReLU(.2)(self.model)
        # self.model = Dropout(.25)(self.model)
        #
        # self.model = Conv2D(128,(3,3),padding='same', strides=(2,2))(self.model)
        # self.model = LeakyReLU(.2)(self.model)
        # # self.model = BatchNormalization()(self.model)
        #
        # self.model = Conv2D(128, (3, 3), padding='same')(self.model)
        # self.model = LeakyReLU(.2)(self.model)
        # self.model = Dropout(.25)(self.model)
        # # self.model = Conv2D(128, (3, 3), padding='same')(self.model)
        # # self.model = LeakyReLU(.2)(self.model)
        #
        # self.model = Conv2D(256,(3,3), padding='same',strides=(2,2))(self.model)
        # self.model = LeakyReLU(.2)(self.model)
        # self.model = Dropout(.25)(self.model)
        # # self.model = BatchNormalization()(self.model)
        #
        # self.model = Flatten()(self.model)
        # self.model = Dense(512)(self.model)
        # self.model = LeakyReLU(.2)(self.model)
        # self.model = Dropout(.5)(self.model)
        # self.model = Dense(1)(self.model)
        # self.model = Activation('sigmoid')(self.model)

    def compile_w_summary(self):
        self.discriminator = Model(self.d_input,self.model)
        opt = Adam(lr=.0001)
        self.discriminator.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
        print('\n')
        print('Discriminator summary...\n')
        print(self.discriminator.summary())
        return self.discriminator

    def compile(self):
        self.discriminator = Model(self.d_input,self.model)
        opt = Adam(lr=.0001)
        self.discriminator.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
        return self.discriminator

    def fit(self, X_train, y_train, X_test, y_test, batch_size=32, epochs=100):
        self.discriminator.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1,validation_data=(X_test,y_test), shuffle=True)

    def evaluate(self, x,y):
        return self.discriminator.evaluate(x=x,y=y)

    def predict(self, X, batch_size=32):
        return self.discriminator.predict(X,batch_size=batch_size, verbose=1)

    def make_trainable(self,val):
        self.discriminator.trainable = val
        for layer in self.discriminator.layers:
            layer.trainable = val

    def train_on_batch(self, X, y):
        return self.discriminator.train_on_batch(X,y)

    def save(self,name):
        self.discriminator.save('../models/' + name +'.h5')

class GAN():
    def compile(self,input_shape, output_shape):
        gan_input = Input(shape=input_shape)
        self.g = Generator()
        self.g.build(input_shape=input_shape)
        self.d = Discriminator()
        self.d.build(input_shape=output_shape)
        self.generator = self.g.compile()
        self.discriminator = self.d.compile_w_summary()
        model = self.g.generator(gan_input)
        gan_V = self.d.discriminator(model)
        self.gan = Model(gan_input,gan_V)
        opt = Adam(lr=.001)
        self.gan.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
        print('\n')
        print('GAN summary...')
        print(self.gan.summary())

    # def d_make_trainable(self, val):
    #     self.d.make_trainable(val)
    #
    # def g_make_trainable(self,val):
    #     self.g._make_trainable(val)

    def train_on_batch(self,X,y):
        return self.gan.train_on_batch(X,y)

    def save_g(self,name):
        self.generator.save('../models/' + name + '.h5')

    def save_d(self,name):
        self.discriminator.save('../models/' + name + '.h5')

    def predict(self,X):
        return self.g.predict(X)
