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
5. [Results](#results)

## What is a GAN

A generative adversarial network or "GAN" is a neural network consisting of two submodels. These models work together to generate something, could be an image or even music, that to humans seems like the "real" thing.

The first submodel is the "Generator" and the second is the "Discriminator." After being pre-trained on what is real and what is noise, the Discriminator trains the Generator by revealing to it when it has created something realistic and when it hasn't. At first, the Generator will produce mostly noise, but eventually it will fool the Discriminator.

There are lots of types of GANs that researchers have given creative names to, such as DCGANs, HyperGans, CycleGANs, and S^2-GANs. Each are tweaked in certain ways to be more suitable to a specific task. However, all share the core principle of one net training the other to generate novel content.

![GAN](/test_images/GAN_arch.jpeg)

## Color

### RGB
Most images use the RGB colorspace. The disadvantage of using RGB when colorizing images is that the model needs to predict 3 values for each pixel.

![RGB](/test_images/rgb.jpg)

### CIE-LAB

This project will be utilizing the CIE-LAB color space to preserve the gray scaled image. As shown in the picture below, the gray scaled images is simply the L value in LAB. Therefore, the generator will use the L spectrum as input and predict A and B. To view the results, the L layer is added back in and LAB had to be converted to RGB.

![CIE](/test_images/cie.png)

## Model Architecture

### Generator

The goal of the Generator is to create content so indistinguishable from the training set that the Discriminator cannot tell the difference.

## Discriminator

## Results

![](/data/Paint/For_readme/red.png =100x100)![](/data/Paint/For_readme/red_gray.png =100x100)![](/test_images/22/for_pres/red.png =100x100)



![D_loss](/images/Generative_Losses_512_24_epochs.png)

![G_loss](/images/Discriminative_Losses_512_24_epochs.png)
