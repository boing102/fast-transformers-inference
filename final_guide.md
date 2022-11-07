# Deploy Triton on Sagemaker

In my labweek, I deployed the Triton inference server on Sagemaker. This was to reproduce the sub 1ms inference on Huggingface transformers models from [this blog post](https://towardsdatascience.com/hugging-face-transformer-inference-under-1-millisecond-latency-e1be0057a51c). 
The blog post author created a [repo](https://github.com/ELS-RD/transformer-deploy/tree/72fff87fdf008b4c3ec73e06534fb022c25da094) with tooling that performs the necessary step to host the models on Triton. We will use the tooling provided and some additional steps to a working Sagemaker Endpoint running the Triton Server with a Huggingface model. 

1. Convert your model into "TensorRT format".
2. Package the model in Sagemaker supported format.
3. Run the Triton server as a Sagemaker Endpoint.


## Convert your model into "TensorRT format"

This is super easy because of the tooling provided from the repo. You'll just need a docker image and that's it. 
I'm using the `sentence-transformers/all-mpnet-base-v2` model in this example.

> **WARNING: You have to run this step on the exact same type of GPU, you're gonna run the inference on!**

```
docker run -it --rm --gpus all \
  -v $PWD:/project ghcr.io/els-rd/transformer-deploy:0.5.1 \
  bash -c "cd /project && \
    convert_model -m \"sentence-transformers/all-mpnet-base-v2\" \
    --backend tensorrt onnx \
    --task embedding \
    --seq-len 2 16 128 \
    -o all-mpnet-base-v2 \
    -n all-mpnet-base-v2"
```

This command first converts the model into ONNX format, then the TensorRT format and prepares both of these models to run on Triton. 
In my understanding, the Triton server will consider this 
particular model as an ensamble. In the `all-mpnet-base-v2/` folder, there are additional folders
like `tensorrt_inference`, `tensorrt_model` & `tensorrt_tokenize` each part of the ensamble.

## Package the model in Sagemaker supported format
The tokenizer submodel is using the Huggingface Tranformers library to tokenize the input. 
This library is not available in the amazon deep learning image we will be using in Sagemaker. 
The Triton server has a buildin support for customer environments using conda. 
We will package our environment as conda env and tell the server to load this on startup.
This process is the same for any python dependecies that the submodels of the ensamble use.

Run the following commands from the folder you ran the model generation command from:
```
cd all-mpnet-base-v2
conda create -y -n all-mpnet-base-v2_tensorrt_tokenize python=3.8
source ~/anaconda3/etc/profile.d/conda.sh
conda activate all-mpnet-base-v2_tensorrt_tokenize
export PYTHONNOUSERSITE=True
conda install -y -c conda-forge transformers
pip install conda-pack
conda-pack
mv all-mpnet-base-v2_tensorrt_tokenize.tar.gz all-mpnet-base-v2_tensorrt_tokenize
```

Now we have to point the Triton server to load this environment on startup, so that the submodel can use the required library.
This is easy using the configuration file of the submodel. Just add
```
parameters: {
  key: "EXECUTION_ENV_PATH",
  value: {string_value: "$$TRITON_MODEL_DIRECTORY/all-mpnet-base-v2_tensorrt_tokenize.tar.gz"}
}
```
to the end of `all-mpnet-base-v2_tensorrt_tokenize/config.pbtxt`.

All that is left to do now is tar this whole folder. From the same folder you ran the previous commands from:

```
tar --exclude='.ipynb_checkpoints' -czvf model.tar.gz .
```
## Run the Triton server as a Sagemaker Endpoint

Now that everything ready, we can deploy the Sagemaker Endpoint. I did this from a Sagemaker notebook, so that I have a role with all the required permissions. I used a sagemaker SDK, but I believe that using the model serving platform is trivial - just change the image and upload the `model.tar.gz` to where model serving expects it.

You can run the code from `triton_inference.py` from a notebook or something.

The requests to the Triton server are not just a plain json. The request needs special headers and the body to be encoded in a special way. 
You can see how to make requests at the end of the `triton_inference.py` file. The request body is based on the `all-mpnet-base-v2_tensorrt_inference/config.pbtxt`.


