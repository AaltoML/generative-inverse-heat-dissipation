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

![](assets/fig/teaser.jpg)
