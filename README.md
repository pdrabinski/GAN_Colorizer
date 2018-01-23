# GANs: Not always Black and White
_________________________________________________________________
Colorizing black and white photos is currently a painstaking and labor-intensive process. It has to be done manually in photoshop by a skilled graphic designer and the whole process can take a long time because it relies on the designer's imagination and efficiency to produce a realistic colorization.

GAN's can circumvent this by developing their own "intuition" over thousands of training iterations. This "intuition" helps them recognize patterns in images and apply the correct colorization. This project seeks to build a GAN in Keras that can accomplish this.

## Table of Contents
1. [What is a GAN?](#what-is-a-gan?)
2. [How to train a GAN?](#how-to-train-a-gan?)
3. [Color Spectrums](#color-spectrums)
    *  [RGB](#rgb)
    *  [CIE-LAB](#cie-lab)
4. [Model Architecture](#model-Architecture)
    *  [Generator](#Generator)
    *  [Discriminator](#Discriminator)
    *  [GAN](#gan)
5. [Results](#results)

## What is a GAN

A generative adversarial network or "GAN" is a neural network consisting of two submodels. These models work together to generate something, could be an image or even music, that to humans seems like the "real" thing.

The first submodel is the "Generator" and the second is the "Discriminator." After being pre-trained on what is real and what is noise, the Discriminator trains the Generator by revealing to it when it has created something realistic and when it hasn't. At first, the Generator will produce mostly noise, but eventually it will fool the Discriminator.

There are lots of types of GANs that researchers have given creative names to, such as DCGANs, HyperGans, CycleGANs, and S^2-GANs. Each are tweaked in certain ways to be more suitable to a specific task. However, all share the core principle of one net training the other to generate novel content.

![GAN](/results/GAN_arch.jpeg)

## Color

### RGB
Most images use the RGB colorspace. The disadvantage of using RGB when colorizing images is that the model needs to predict 3 values for each pixel.

![RGB](/results/rgb.jpg)

### CIE-LAB

This project will be utilizing the CIE-LAB color space to preserve the gray scaled image. As shown in the picture below, the gray scaled images is simply the L value in LAB. Therefore, the generator will use the L spectrum as input and predict A and B. To view the results, the L layer is added back in and LAB had to be converted to RGB.

![CIE](/results/cie.png)

## Model Architecture

### Generator

The goal of the Generator is to create content so indistinguishable from the training set that the Discriminator cannot tell the difference.

![generator](/results/generator.png)

### Discriminator

The goal of the Discriminator is to be the expert on what a true image looks like. If it is fooled by the Discriminator too early then it is not doing its job well enough and as a result, will not be able to train the Generator well.

Image here.

### GAN

GAN summary...
_________________________________________________________________
Layer (type)&ensp;&ensp;&ensp;&ensp;&ensp;Output Shape&ensp;&ensp;&ensp;&ensp;Params
_________________________________________________________________
Generator&ensp;&ensp;&ensp;&ensp;(None, 32, 32, 1)&ensp;&ensp;&ensp;&ensp;0
_________________________________________________________________
Discriminator&ensp;&ensp;&ensp;&ensp;&ensp;(None, 32, 32, 2)&ensp;&ensp;&ensp;&ensp;205794
_________________________________________________________________
sequential_1 (Sequential)&ensp;&ensp;&ensp;&ensp;(None, 1)&ensp;&ensp;&ensp;&ensp;23585
_________________________________________________________________
Total params: 229,379
Trainable params: 205,154
Non-trainable params: 24,225

## Results

Ground Truth | Grayscaled Image | Colorized

<img src="/data/Paint/For_readme/red.png" width="128" /><img src="/data/Paint/For_readme/red_gray.png" width="128"/><img src="/results/22/for_pres/red.png" width="128" />
<br>
<img src="/data/Paint/For_readme/blue.png" width="128" /><img src="/data/Paint/For_readme/blue_gray.png" width="128"/><img src="/results/22/for_pres/blue.png" width="128" />


![D_loss](/plots/Plots/generative_plot.png)

![G_loss](/plots/Plots/discriminative_plot.png)

#### How does it do on a more complex example?
<img src="/results/22/for_pres/sailboat_color.png" width="128" /><img src="/results/22/for_pres/sailboat_gray.png" width="128"/><img src="/results/22/for_pres/sailboat.png" width="128" />
