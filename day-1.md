# What I did

1. Downloaded the image responsible for converting a model to the fast format & clone repo https://github.com/ELS-RD/transformer-deploy
It didn't work on the gpu docker machines, becasue there was too litle space. Just spinned up P3 instance in staging with the `Deep Learning AMI GPU PyTorch 1.12.1 (Ubuntu 20.04) 20221025`.

2. Convert the model.
```
docker run -it --rm --gpus all \
  -v $PWD:/project ghcr.io/els-rd/transformer-deploy:0.5.1 \
  bash -c "cd /project && \
    convert_model -m \"facebook/contriever-msmarco\" \
    --backend tensorrt onnx \
    --task embedding \
    --seq-len 4 128 128 \
    --output contriever-msmarco"
```

The `seq-len` is minimum, optimal, maximum input sequence length, to help TensorRT better optimize your model. [Docs](https://els-rd.github.io/transformer-deploy/run/)

Some warnings when converting to TensorRT model:
```
Weights [name=model.0.auto_model.embeddings.word_embeddings.weight] had the following issues when converted to FP16:
[10/31/2022-11:31:08] [TRT] [W]  - Subnormal FP16 values detected.
[10/31/2022-11:31:08] [TRT] [W]  - Values less than smallest positive FP16 Subnormal value detected. Converting to FP16 minimum subnormalized value.
[10/31/2022-11:31:08] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[10/31/2022-11:31:08] [TRT] [W] Weights [name=(Unnamed Layer* 1893) [Constant] + (Unnamed Layer* 1895) [Shuffle]] had the following issues when converted to FP16:
[10/31/2022-11:31:08] [TRT] [W]  - Finite FP32 values which would overflow in FP16 detected.  Converting to closest finite FP16 value.
[10/31/2022-11:31:08] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[10/31/2022-11:31:08] [TRT] [W] Weights [name=model.0.auto_model.embeddings.position_embeddings.weight] had the following issues when converted to FP16:
[10/31/2022-11:31:08] [TRT] [W]  - Subnormal FP16 values detected.
[10/31/2022-11:31:08] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
```
Todo: invatigate...

Now i have `multi-qa-mpnet-base-dot-v1` & `all-mpnet-base-v2`.

3. Run the triton server & query on EC2

```
docker run -it --rm --gpus all -p8000:8000 -p8001:8001 -p8002:8002 --shm-size 256m \
  -v $PWD/all-mpnet-base-v2:/models nvcr.io/nvidia/tritonserver:22.07-py3 \
  bash -c "pip install transformers && tritonserver --model-repository=/models"
```

```
curl -X POST  http://localhost:8000/v2/models/transformer_onnx_inference/versions/1/infer \
  --data-binary "@demo/infinity/query_body.bin" \
  --header "Inference-Header-Content-Length: 161"
```

Also, using apache bench:
```
ab -H "Inference-Header-Content-Length: 161" -H "Content-Length: 220" -p "demo/infinity/query_body.bin" -T "application/x-www-form-urlencoded" -n 2000 -c 2 http://localhost:8000/v2/models/transformer_tensorrt_inference/versions/1/infer
ab -H "Inference-Header-Content-Length: 161" -H "Content-Length: 186" -p "body.bin" -T "application/x-www-form-urlencoded" -n 2000 -c 2 http://localhost:8000/v2/models/transformer_tensorrt_inference/versions/1/infer
```
To get a nice explanation of what the output look like see [DataDog blog](https://www.datadoghq.com/blog/apachebench/)

If we wanna query using a specific query, we have to convert it to the desired input format according to [this guide](https://github.com/ELS-RD/transformer-deploy/blob/72fff87fdf008b4c3ec73e06534fb022c25da094/docs/run.md#query-the-inference-server).

Interesting query `Simple` is slower than `Simple Simple Simple`.


**First measurements**
| model                       | Triton GPU+Tensor RT, LQ                 | Triton GPU+Tensor RT, SQ                   | Triton CPU+ONNX, SQ  |   |   |
|-----------------------------|------------------------------------------|---------------------------------------------------|---|---|---|
| all-mpnet-base-v2           | 733.70 rps, 2.726ms☆ latency (1.363ms ★) | 767.10 rps, 2.607ms☆ latency (1.304ms ★)| 69.21 rps, 28.898ms☆ latency (14.449ms ★)  |   |   |
| multi-qa-mpnet-base-dot-v1  | 756.15 rps, 2.645ms☆ latency (1.322ms ★) | 784.34 rps, 2.550ms☆ latency (1.275ms ★) | 69.29 rps, 28.864ms☆ latency (14.432ms ★)  |   |   |
```
LQ - Long query
SQ - Short query
★ - (mean, across all concurrent requests)
☆ - mean latency
```
The benchmark was with 2000 requests & concurrency=2. On the `p3.2xlarge` EC2 instance.



Misc. useful commands:
```
docker run -it --rm --gpus all \
  -v $PWD:/project ghcr.io/els-rd/transformer-deploy:0.5.1 \
  bash -c "cd /project && \
    convert_model -m \"sentence-transformers/multi-qa-mpnet-base-dot-v1\" \
    --backend onnx \
    --task embedding \
    --seq-len 4 128 128 \
    -d cpu \
    -v \
    --fast \
    --nb-measures 10\
    --output multi-qa-mpnet-base-dot-v1-cpu"
```

