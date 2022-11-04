# Day 4

Today, let's try to generate the onnx & tensor rt model in similar way like they do in [notebook](https://github.com/aws/amazon-sagemaker-examples/blob/main/sagemaker-triton/nlp_bert/triton_nlp_bert.ipynb) so that we can deploy it easily to sagemaker.

To generate a model:
```
convert_model -m "sentence-transformers/all-mpnet-base-v2" --backend onnx --task embedding --seq-len 2 16 128 -o all-mpnet-base-v2 -n all-mpnet-base-v2
```
This generates the model artifacts to `all-mpnet-base-v2` folder. 


After you generate your optimized model artifacts you have to put them to this specific folder structure:
```
root
 └── all-mpnet-base-v2 # This is the model name 
 	└── config.pbtxt # Take this from all-mpnet-base-v2/all-mpnet-base-v2_tensorrt_inference/
 	└── 1
        └── model.plan # Take this from all-mpnet-base-v2/
```

Tar this folder with:
```
tar -C root/ -czf model.tar.gz all-mpnet-base-v2
```
This is how sagemaker expects it and we will upload this to s3.

Figure out how to stracture sagemaker model using https://raw.githubusercontent.com/triton-inference-server/server/main/docker/sagemaker/serve
