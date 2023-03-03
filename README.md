# Generative Modeling With Inverse Heat Dissipation

## Abstract

While diffusion models have shown great success in image generation, their noise-inverting generative process does not explicitly consider the structure of images, such as their inherent multi-scale nature. Inspired by diffusion models and the empirical success of coarse-to-fine modelling, we propose a new diffusion-like model that generates images through stochastically reversing the heat equation, a PDE that locally erases fine-scale information when run over the 2D plane of the image. We interpret the solution of the forward heat equation with constant additive noise as a variational approximation in the diffusion latent variable model. Our new model shows emergent qualitative properties not seen in standard diffusion models, such as disentanglement of overall colour and shape in images. Spectral analysis on natural images highlights connections to diffusion models and reveals an implicit coarse-to-fine inductive bias in them.

<p align="center">
<img src="assets/fig/teaser.png" alt="drawing" width="70%" style="text-align: center;"/>
</p>
  
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

### LSUN-Churches 128x128

<video width="49%" controls>
  <source src="assets/video/lsun_church_1.mp4" type="video/mp4">
  Your browser does not support the video tag. Download the video <a href="assets/video/lsun_church_1.mp4">here</a>.
</video>
<video width="49%" controls>
  <source src="assets/video/lsun_church_2.mp4" type="video/mp4">
  Your browser does not support the video tag. Download the video <a href="assets/video/lsun_church_2.mp4">here</a>.
</video>

### Interpolations

Qualitatively, interpolations in the full latent space of the model are smoother than the corresponding interpolations in a standard diffusion model. This is also the case when comparing interpolations using DDIM, a deterministic sampler. 

<video width="70% controls align="center>
    <source src="assets/video/interpolations.mp4" type="video/mp4">
Your browser does not support the video tag. Download the video <a href="assets/video/interpolations.mp4">here</a>.
</video>

### Sampling with a shared initial state

<video width="70%" controls align="center">
  <source src="assets/video/mnist_hierarchical.mp4" type="video/mp4">
  Your browser does not support the video tag. Download the video <a href="assets/video/mnist_hierarchical.mp4">here</a>.
</video>

<video width="70%" controls align="center">
  <source src="assets/video/afhq64_sameinit_1.mp4" type="video/mp4">
  Your browser does not support the video tag. Download the video <a href="assets/video/afhq64_sameinit_1.mp4">here</a>.
</video>

<video width="70%" controls align="center">
  <source src="assets/video/afhq64_sameinit_2.mp4" type="video/mp4">
  Your browser does not support the video tag. Download the video <a href="assets/video/afhq64_sameinit_2.mp4">here</a>.
</video>
