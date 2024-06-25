
# Installation
If you are installing the repository as a standalone, follow the instructions in the [README_original.md](README_original.md) in the root directory of this repository.
If you are installing this as a submodule, follow the instructions in the main README of the root directory of the parent repository.


# Overview

Following is the structure of the codebase learnt using debugging and reading the code.

## Graph

The base class that is an interface to perform forward noising, and some utility functions that are useful while denoising that depend on forward noising.

## Noise
Just a variance/noise level scheduler that given t, and the total number of steps, returns the parameters that depend on the noise level like the variance of the noise.  


## Sampling

The `run_sample.py` file contains the code to sample. 
The [get_pc_sampler](https://github.com/dhruvdcoder/Score-Entropy-Discrete-Diffusion/blob/0605786da5ccb5747545e26d66fdf477187598b6/sampling.py#L122) creates a sampler object.

### Predictor
The [Predictor](https://github.com/dhruvdcoder/Score-Entropy-Discrete-Diffusion/blob/0605786da5ccb5747545e26d66fdf477187598b6/sampling.py#L38) is the base predictor class that takes x, t, and score_fn as input and returns the x for the next time step t-1. There are three types of predictors: "none", "euler"  and ["analytic"](https://github.com/dhruvdcoder/Score-Entropy-Discrete-Diffusion/blob/0605786da5ccb5747545e26d66fdf477187598b6/sampling.py#L78) (default).


### Denoisor
Seems to be doing the same thing as "analytic" predictor. Ahh. The only thing different is that it removes the MASK token from the vocabulary.
So the denoisor is called once at the end of the inference process to produce a sentence without the MASK token. I don't think this is mentioned in the paper.



# Analysis of intermediate samples

I looked at the intermediate outputs of the inference process (unconditional generation) for various step sizes. The results are in `exp_local/intermediate`.
To run the analysis for yourself (remove the prefix directory SEDD, if you are using this repo as a standalone and not running from the parent):
```
mkdir -p SEDD/exp_local/intermediate
python SEDD/run_sample.py --max-length 100 --steps 10 --log-file SEDD/exp_local/intermediate/outputs_100_10.json
module load ffmpeg/4.4.1 # ensure you have ffmpeg on path
python SEDD/animate.py --input SEDD/exp_local/intermediate/outputs_100_10.json
```