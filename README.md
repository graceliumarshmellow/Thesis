# Thesis
Here is most of the code used for my thesis on using deep learning to improve the quality of luminescence images.
Some of the code requires to be run on a GPU and some require you to provide your own data.

See the README BM3D for the credits of the BM3D code. BM3D.py was modified for this project.

- CalculatePSNR&SSIM.ipynb - code used to calculate the PSNR and SSIM of an image
- ELImagesCode.ipynb - code used to train the U-net and run the BM3D code
- AddPoissonNoise.m - code used to add Poisson noise to the clean images. Note that noisy images used for the project had Gaussian and Poisson noise. It is just Gaussian was added via a method shown in ELImagesCode.ipynb
