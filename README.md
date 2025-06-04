# Conditional Super Resolution VAE

Code repo of my internship subject at [ISAE-SUPAERO](https://www.isae-supaero.fr/)

Inspired by [1] I. Gatopoulos, M. Stol, et J. M. Tomczak, « Super-resolution variational auto-encoders », arXiv preprint arXiv:2006.05218, 2020, [Online] Available at : https://arxiv.org/pdf/2006.05218

This repository contains an implementation of the Conditional VAE applied to Super Resolution of Satellite Images.
Its short to mid-term prospect is to quantify uncertainties on the super resolution task.

## Particularities

For now, the code is modified to use Gaussian priors and decoders for simplicity.

*__Note__*: In grid mode, the batch size depends on the crop size (using a 64px\*64px crop, it yields 16 patches for one 256 image so a batch is 16\*batch_size)
