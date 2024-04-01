**Project AN2 - Disentanglement in variational autoencoders**

This project aims at studying disentangled representations of the dSprites dataset given diverse generative factors (shape of the image, scale, positionX, positionY, orientation), using two diverse approaches :

- BetaVAE : adds a hyperparameter $\beta$ to encourage independence of factors in the latent space
- FactorVAE : adds a term in the expression of the ELBO to encourage a factorised distribution in the latent space

We propose the following series of experiments in our project :
 - Reimplementation of both architectures
 - Comparison of their reconstruction power
 - Qualitative study of disentanglement : latent traversals
 - Implementation of the metric score of both models

*Teddy ALEXANDRE, Gabrielle LE BELLIER / DeLIReS - MVA Master 2023-2024*
