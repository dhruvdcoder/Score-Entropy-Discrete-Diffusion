
1. Create a fresh conda env:

```
conda create -p ./.venv_sedd python=3.9.7 pip ipykernel
```

2. Install the dependencies as specified by the original authors

```
conda activate ./.venv_sedd
```
# Graph

The base class that is an interface to perform forward noising, and some utility functions that are useful while denoising that depend on forward noising.

# Noise
Just a variance/noise level scheduler that given t, and the total number of steps, returns the parameters that depend on the noise level like the variance of the noise.  


# Sampling

The `run_sample.py` file contains the code to sample. 
The [get_pc_sampler](https://github.com/dhruvdcoder/Score-Entropy-Discrete-Diffusion/blob/0605786da5ccb5747545e26d66fdf477187598b6/sampling.py#L122) creates a sampler object.

## Predictor
The [Predictor](https://github.com/dhruvdcoder/Score-Entropy-Discrete-Diffusion/blob/0605786da5ccb5747545e26d66fdf477187598b6/sampling.py#L38) is the base predictor class that takes x, t, and score_fn as input and returns the x for the next time step t-1. There are three types of predictors: "none", "euler"  and ["analytic"](https://github.com/dhruvdcoder/Score-Entropy-Discrete-Diffusion/blob/0605786da5ccb5747545e26d66fdf477187598b6/sampling.py#L78) (default).


## Denoisor
Seems to be doing the same thing as "analytic" predictor. Ahh. The only thing different is that it removes the MASK token from the vocabulary.
So the denoisor is called once at the end of the inference process to produce a sentence without the MASK token. I don't think this is mentioned in the paper.



# Analysis of intermediate samples

I looked at the intermediate outputs of the inference process (unconditional generation) for various step sizes. The results are in `exp_local/intermediate`.