
*This website contains sample trajectory visualizations for our paper "Generative Modeling With Inverse Heat Dissipation".*

While diffusion models have shown great success in image generation, their noise-inverting generative process does not explicitly consider the inductive biases of natural images, such as their inherent multi-scale nature. Inspired by diffusion models and the desirability of coarse-to-fine modelling, we propose a new model that generates images through iteratively inverting the heat equation, a PDE that locally erases fine-scale information when run over the 2D plane of the image. In our novel methodology, the solution of the forward heat equation is interpreted as a variational approximation in a directed graphical model. We demonstrate promising image quality and point out emergent qualitative properties not seen in diffusion models, such as disentanglement of overall colour and shape in images and aspects of neural network interpretability. Spectral analysis on natural images positions our model as a type of dual to diffusion models.

<p align="center">
<img src="assets/fig/teaser.png" alt="" width="70%" style="text-align: center;"/>
</p>
  
## Sample trajectories

The iterative generative process can be visualized as a video, showing the smooth change from effective low-resolution to high resolution. Effectively, the model redistributes the mass in the original image to form an image.

### MNIST

<p align="center">
<video width="49%" controls>
  <source src="assets/video/mnist.mp4" type="video/mp4">
  Your browser does not support the video tag. Download the video <a href="assets/video/mnist.mp4">here</a>.
</video>
</p>

### CIFAR-10

<p align="center">
<video width="49%" controls>
  <source src="assets/video/cifar10.mp4" type="video/mp4">
  Your browser does not support the video tag. Download the video <a href="assets/video/cifar10.mp4">here</a>.
</video>
</p>
  
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

### Sampling with a shared initial state

One way to visualize the stochasticity of the generative process is to keep the initial draw from the prior fixed and sample multiple trajectories based on it. Large-scale features are decided in the beginning of the process and fine-scale features at the end. If we split the sampling to two parts at specified moments, this results in a hierarchy over scales:

<p align="center">
<video width="70%" controls align="center">
  <source src="assets/video/mnist_hierarchical.mp4" type="video/mp4">
  Your browser does not support the video tag. Download the video <a href="assets/video/mnist_hierarchical.mp4">here</a>.
</video>
</p>

Starting from the same initial state results in a wide variety of images. Here are examples from a low-resolution version of AFHQ (64x64):

<p align="center">
<video width="70%" controls align="center">
  <source src="assets/video/afhq64_sameinit_1_a.mp4" type="video/mp4">
  Your browser does not support the video tag. Download the video <a href="assets/video/afhq64_sameinit_1_a.mp4">here</a>.
</video>
</p>
  
<p align="center">
<video width="70%" controls align="center">
  <source src="assets/video/afhq64_sameinit_2_a.mp4" type="video/mp4">
  Your browser does not support the video tag. Download the video <a href="assets/video/afhq64_sameinit_2_a.mp4">here</a>.
</video>
</p>
