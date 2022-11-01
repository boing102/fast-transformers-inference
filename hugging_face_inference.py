from sagemaker.huggingface import HuggingFaceModel
import sagemaker 

role = sagemaker.get_execution_role()

# Hub Model configuration. https://huggingface.co/models
hub = {
  'HF_MODEL_ID':'sentence-transformers/multi-qa-mpnet-base-dot-v1', # model_id from hf.co/models
  'HF_TASK':'sentence-similarity' # NLP task you want to use for predictions
}

# create Hugging Face Model Class
huggingface_model = HuggingFaceModel(
   env=hub,
   role=role, # iam role with permissions to create an Endpoint
   transformers_version="4.17", # transformers version used
   pytorch_version="1.10.2", # pytorch version used
   py_version="py38", # python version of the DLC
)



# deploy model to SageMaker Inference
predictor = huggingface_model.deploy(
   initial_instance_count=1,
   instance_type="ml.c5.large"
)


