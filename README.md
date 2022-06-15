# Generative Modeling With Inverse Heat Dissipation

## Abstract

While diffusion models have shown great success in image generation, their noise-
inverting generative process does not explicitly consider the inductive biases of
natural images, such as their inherent multi-scale nature. Inspired by diffusion
models and the desirability of coarse-to-fine modelling, we propose a new model
that generates images through iteratively inverting the heat equation, a PDE that
locally erases fine-scale information when run over the 2D plane of the image. In
our novel methodology, the solution of the forward heat equation is interpreted
as a variational approximation in a directed graphical model. We demonstrate
promising image quality and point out emergent qualitative properties not seen in
diffusion models, such as disentanglement of overall colour and shape in images
and aspects of neural network interpretability. Spectral analysis on natural images
positions our model as a type of dual to diffusion models.

![center](assets/fig/teaser.png){ width=50% }

[<img src="assets/fig/teaser.png" width="70%"/>](assets/fig/teaser.png)

<img src="assets/fig/teaser.png" alt="drawing" width="70%" class="center"/>

## Sample trajectories

### MNIST

<video width="49%" controls>
  <source src="assets/video/mnist.mp4" type="video/mp4">
  Your browser does not support the video tag. Download the video <a href="assets/video/mnist.mp4">here</a>.
</video>

### CIFAR-10

<video width="49%" controls>
  <source src="assets/video/cifar10.mp4" type="video/mp4">
  Your browser does not support the video tag. Download the video <a href="assets/video/cifar10.mp4">here</a>.
</video>

### AFHQ 256x256

<video width="49%" controls>
  <source src="assets/video/afhq_1.mp4" type="video/mp4">
  Your browser does not support the video tag. Download the video <a href="assets/video/afhq_1.mp4">here</a>.
</video>
<video width="49%" controls>
  <source src="assets/video/afhq_2.mp4" type="video/mp4">
  Your browser does not support the video tag. Download the video <a href="assets/video/afhq_2.mp4">here</a>.
</video>

### FFHQ 256x256

<video width="49%" controls>
  <source src="assets/video/ffhq_1.mp4" type="video/mp4">
  Your browser does not support the video tag. Download the video <a href="assets/video/ffhq_1.mp4">here</a>.
</video>
<video width="49%" controls>
  <source src="assets/video/ffhq_2.mp4" type="video/mp4">
  Your browser does not support the video tag. Download the video <a href="assets/video/ffhq_2.mp4">here</a>.
</video>

### Sampling with a shared initial state

<video width="70%" controls>
  <source src="assets/video/afhq64_sameinit_1.mp4" type="video/mp4">
  Your browser does not support the video tag. Download the video <a href="assets/video/afhq64_sameinit_1.mp4">here</a>.
</video>
