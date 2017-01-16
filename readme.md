A Generative Restricted Boltzmann Machine with binary visible and hidden units.

Training can be done by Contrastive Divergence or Persistent Contrastive Divergence. Additional hyperparameters such as weight decay constant and the momentum can be fine-tuned for best result.

After training, the model can classify unlabelled data and reconstruct unseen data with the learned distribution.

## Result on MNIST

![reconstructed mnist](https://github.com/michaelzhou0723/rbm/blob/master/demos/figure.png)
