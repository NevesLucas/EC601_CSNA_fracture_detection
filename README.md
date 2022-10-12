# EC601_CSNA_fracture_detection
Class project for detection of Spinal Fractures in CT Data

# Conda Environment Setup

To setup a conda environment with required dependencies, first install your favorite flavor of anaconda.
- For miniconda: [miniconda](https://docs.conda.io/en/latest/miniconda.html)
- For full Anaconda: [Anaconda](https://www.anaconda.com/products/distribution)


## Configure environment.

Choose the `environment-XX.yml` file that best matches your system configuration.

- For computers with a CUDA capable GPU, this is `environment-CUDA.yml`
- For any Windows PC running windows 10 or later, this is `environment-WIN.yml`
- For anything not listed above (mac included), use `environment-CPU.yml`

then run the configuration to create a conda env:

```
conda env create -f YOUR_CONFIG.yml
```
The environment will be created according to the configuration chosen, and can then be activated with:

```
conda activate csna
```

# Net selection

We are tring to use UNet model to complete our project.
