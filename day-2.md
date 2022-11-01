# Day 2

Deploy actual endpoints to sagemaker and test the latency+ throughput there...

- [ ]  Vanilla model
- [ ] Triton CPU + ONNX
- [ ] Triton GPU + ONNX
- [ ] Triton GPU + Tensor RT

How to easily create sagemaker endpoints?

Hugging face has nice [script](https://github.com/huggingface/notebooks/blob/main/sagemaker/11_deploy_model_from_hf_hub/deploy_transformer_model_from_hf_hub.ipynb) But what about the role & credentials?

FML, spent an hour trying to figure out the credentials from inide the EC2 machine, with assumed creds from the mac. Nothing worked. 

Turns out, it's the easiest to just create a sagemaker notebook and choose create new IAM role in the start wizard... (Don't forget to give it permissions for s3 buckets, I just gave to all s3 buckets.)

Making request to deployed endpoint with curl (this also took a while to figure out :D...):

```
curl --request POST "https://runtime.sagemaker.eu-central-1.amazonaws.com/endpoints/huggingface-pytorch-inference-2022-11-01-12-56-36-522/invocations" \
--user "$ACCESS_KEY:$SECRET_KEY" \
--aws-sigv4 "aws:amz:eu-central-1:sagemaker" \
-H "x-amz-security-token:$SECURITY_TOKEN" \
-H "Content-Type: application/json" \
--data '{"inputs":"This is a sentence"}' \  # This depends on the model ofc...
--verbose
```

Then finally run a benchmark on the actual sagemaker endpoint:

```
ab -p "post.json" -T "application/json" -n 1 -c 1 -v 3 \
-H "x-amz-security-token:$SECURITY_TOKEN" \
-H "Host:runtime.sagemaker.eu-central-1.amazonaws.com" \
-H "X-Amz-Date:20221101T160553Z" \
-H "Authorization:???" \
https://runtime.sagemaker.eu-central-1.amazonaws.com/endpoints/huggingface-pytorch-inference-2022-11-01-12-56-36-522/invocations
```

The above doesn't work. Have to investigate further... :D 