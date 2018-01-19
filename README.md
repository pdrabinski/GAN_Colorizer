# GANs: Not always Black and White
_________________________________________________________________
Using a generative adversarial network to "hallucinate" color in black and white photos.

Work in progress.

## What is a GAN

A generative adversarial network or "GAN" is a neural network consisting of two submodels. These models work together to generate something, could be an image or even music, that to humans seems like the "real" thing.

The first submodel is the "Generator" and the second is the "Discriminator." After being pre-trained on what is real and what is noise, the Discriminator trains the Generator by revealing to it when it has created something realistic and when it hasn't. At first, the Generator will produce mostly noise, but eventually it will fool the Discriminator.

![GAN](/test_images/GAN_arch.jpeg)

## Project Goal

Colorizing black and white photos is currently a painstaking and labor-intensive process. It has to be done manually in photoshop by a skilled graphic designer and the whole process can take a long time because it relies on the designer's imagination and efficiency to produce a realistic colorization.

GAN's can circumvent this by developing their own "intuition" over thousands of training iterations that research has shown is remarkably accurate. I would like to replicate these studies, but with as simple a Keras model as will still produce accurate results. Personally, I hope to learn how GANs train themselves, how Convolutional and Deconvolutonal layers can work together, and about Keras.

## Color

### RGB
Most images use the RGB colorspace.

![RGB](/test_images/rgb.jpg | width=500)

### CIE-LAB

This project will be utilizing the CIE-LAB color space to preserve the gray scaled image.

![CIE](/test_images/cie.png)

## Initial Results

####  GAN summary
_________________________________________________________________
###### Layer (type)&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;Output Shape&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;Params
_________________________________________________________________
input_1 (InputLayer)&nbsp;&nbsp;&nbsp;&ensp;(None, 32, 32, 1) &ensp;&ensp;&ensp;&ensp;&ensp;&ensp;0
_________________________________________________________________
model_1 (Model)&ensp;&ensp;&ensp;&ensp;&ensp;(None, 32, 32, 2)&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;380930
_________________________________________________________________
model_2 (Model) &ensp;&ensp;&ensp;&ensp;(None, 1)&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;2163457
_________________________________________________________________
Total params: 2,544,387 <br>
Trainable params: 2,543,555<br>
Non-trainable params: 832<br>
_________________________________________________________________

![Inital results](/test_images/18/forpres.png)

![Inital results](/test_images/19/screenshot.png)

![D_loss](/images/Generative_Losses_512_24_epochs.png)

![G_loss](/images/Discriminative_Losses_512_24_epochs.png)


## Next Steps

Priorities:
1.  Prevent Discriminator from overfitting
2.  Improve discriminator
3.  Create simpler dataset

Things to investigate:
1.  More balanced training algorithm
2.  Larger images or simpler
3.  more specific dataset
4.  effect of dropouts in generator
5.  effect of adjusting the learning rate
6.  different activation functions
7.  image pixels between -1 and 1 (with tanh activation)
