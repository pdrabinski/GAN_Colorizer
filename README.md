# GANs: Not always Black and White

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
5. [Data](*data)
6. [Results](#results)
7. [Tech Stack](#tech-stack)

## What is a GAN

A generative adversarial network or "GAN" is a neural network consisting of two submodels. These models work together to generate something, could be an image or even music, that to humans seems like the "real" thing.

The first submodel is the "Generator" and the second is the "Discriminator." After being pre-trained on what is real and what is noise, the Discriminator trains the Generator by revealing to it when it has created something realistic and when it hasn't. At first, the Generator will produce mostly noise, but eventually it will fool the Discriminator.

There are lots of types of GANs that researchers have given creative names to, such as DCGANs, HyperGans, CycleGANs, and S^2-GANs. Each are tweaked in certain ways to be more suitable to a specific task. However, all share the core principle of one net training the other to generate novel content.

<img src="/images/GAN_arch.png" align="center"/>

## How to Train a GAN

Training GANs are a complex operation and there are ongoing debates about the best methods to accomplish this. This is because the neural networks work together and so more often than not you can see when training is going wrong rather than when training can be stopped because the colorizations are accurate. Below is a graph showing generator loss in blue and discriminator accuracy in red.

<img src="/plots/Plots/plot_30_epochs.png" align="center"/>

The goal of training is to keep the discriminator accuracy near 100% and make sure that the generator loss doesn't drop to 0. If the generator loss drops to 0 then it is fooling the discriminator with bad colorizations.

## Color

### RGB
Most images use the RGB colorspace. The disadvantage of using RGB when colorizing images is that the model needs to predict 3 values for each pixel.

<img src="/images/rgb.jpg" width=360 style="float:middle" />

### CIE-LAB

This project will be utilizing the CIE-LAB color space to preserve the gray scaled image. As shown in the picture below, the gray scaled images is simply the L value in LAB. Therefore, the generator will use the L spectrum as input and predict A and B. To view the results, the L layer is added back in and LAB had to be converted to RGB.

<img src="/images/cie.png" width=360 style="float:center" />

## Model Architecture

### Generator

The goal of the Generator is to create content so indistinguishable from the training set that the Discriminator cannot tell the difference.

Below is the basic structure of the Generator. Each encoding layer is a Convolution layer with stride 2 followed by a Batch Normalization layer and then a Leaky ReLU activation layer of slope .2. Each decoding layer is a an Upsampling layer followed by a convolution layer, Batch Normalization, and finally the concatenation layer.

The arrows illustrate how the early layers are concatenated with the later layers. These concatenations help preserve the structure of prominent edges that the decoding layers identified. They are called skip connections and are prevalent when using a neural network to find the mapping between an input image and the output image.

<img src="/images/generator.png" style="float:center" />

### Discriminator

The goal of the Discriminator is to be the expert on what a true image looks like. If it is fooled by the Discriminator too early then it is not doing its job well enough and as a result, will not be able to train the Generator well.

<img src="/images/discriminator.png" />

### GAN

Here is a summary of the overall GAN architecture.

|Layer           |Output Shape     | Params      |
|----------------|-----------------|-------------|
|Input           |(256, 256, 1)    |0            |
|Generator       |(256, 256, 2)    |4,729,922    |
|Discriminator   |(1)              |455,457      |

Total params: 5,185,379<br>
Trainable params: 4,726,082<br>
Non-trainable params: 459,297<br>

## Data

I used two datasets for this project. The first was of my own making. For simplicity, I created simple images of shapes, where each shape is a different color.

The second dataset was a subset of the Computational Visual Cognition Laboratory Urban and Natural Scenes dataset. I trained my model on the forest category first and then once the model was producing fairly good results, I tried the mountain category.

Images: http://cvcl.mit.edu/database.htm

## Results

Ground Truth | Grayscaled Image | Colorized

<img src="/data/Paint/For_readme/red.png" width="128" /><img src="/data/Paint/For_readme/red_gray.png" width="128"/><img src="/images/22/for_pres/red.png" width="128" />
<br>
<img src="/data/Paint/For_readme/blue.png" width="128" /><img src="/data/Paint/For_readme/blue_gray.png" width="128"/><img src="/images/22/for_pres/blue.png" width="128" />

#### How does it do on a more complex example? <br>

##### Forest Landscape
Ground Truth | Grayscaled Image | Colorized<br><br>
<img src="/images/29/for_pres/forest_true.png" /><img src="/images/29/for_pres/forest_gray.png" /><img src="/images/29/for_pres/forest_pred.png" />

##### Mountain Landscape
Ground Truth | Grayscaled Image | Colorized<br><br>
<img src="/images/29/for_pres/mountain_true.png" /><img src="/images/29/for_pres/mountain_gray.png" /><img src="/images/29/for_pres/mountain_pred.png" />
<img src="/images/29/for_pres/lake_true.png" /><img src="/images/29/for_pres/lake_gray.png" /><img src="/images/29/for_pres/lake_pred.png" />
<img src="/images/29/for_pres/peak_true.png" /><img src="/images/29/for_pres/peak_gray.png" /><img src="/images/29/for_pres/peak_pred.png" />

## Tech Stack
<img src="images/tech_stack_banner.png" />

## References

1. [Computational Visual Cognition Laboratory](http://cvcl.mit.edu/database.htm)
2. [Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/pdf/1611.07004.pdf)
3. [Colorful Image Colorization](https://arxiv.org/pdf/1603.08511.pdf)
