# Day 5

Almost there. I found out that the directory structure described in day 4 is not complete. We have to bundle the preprocessing model dependencies.
That is described in detail [here](https://github.com/triton-inference-server/python_backend#2-packaging-the-conda-environment).

TLDR of that article is that you if a triton model is using a python backend, you can package the environment as a conda environment. 
In our case the python model is the tokenizer, that uses `transformers`. 
So we have to create an environment where we have the `transfomers` python package.
To do this:

```
conda create -y -n all-mpnet-base-v2_tensorrt_tokenize python=3.8
source ~/anaconda3/etc/profile.d/conda.sh
conda activate all-mpnet-base-v2_tensorrt_tokenize
export PYTHONNOUSERSITE=True
conda install -y -c conda-forge transformers
pip install conda-pack
conda-pack
```

This will produce a `all-mpnet-base-v2_tensorrt_tokenize.tar.gz`.
