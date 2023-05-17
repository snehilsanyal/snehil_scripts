# Script for loading a local model to interact with langchain
import argparse 
import os 
import numpy as np
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from peft import PeftModel 
from langchain import PromptTemplate, LLMChain
from langchain.llms import HuggingFacePipeline

parser = argparse.ArgumentParser(description = 'Load locally hosted model from server\
                                 for further usage via LangChain')
parser.add_argument('-m', '--model_name', description = 'Model name')
parser.add_argument('-q', '--question', description = 'Question from user')
args = parser.parse_args()

# Model path (directory folder) contains pytorch.bin/flaxweights/model.safetensors and tokenizer
llm_dir = '/root/models/llm/'
base_models = ['bloom-1b1', 'bloom-7b1']
model_dict = {'bloom-1b1': ['base', 'lora', 'lora-r128'], 'bloom-7b1': ['base']}

# Return models related to bloom-1b1 (Later for further automation)
# [model_name for model_name in model_names if 'bloom-1b1' in model_name]

# Input from user
model_name = args.model_name
# Check what is the base model name for the input model
idx = np.where([model_name.startswith(base_model) for base_model in base_models])[0].item()
# This is the base model name
base_model_name = base_models[idx]
base_model_path = llm_dir + base_model_name

## PEFT tuners
# Check whether input model is LORA model or base_model
is_lora = ('lora' in model_name)
is_base = (not is_lora) or ('base' in model_name)

# model_path can be base or peft (general)
model_path = llm_dir + model_name 
filelist = os.listdir(model_path)

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(base_model_path, return_dict = True, device_map = 'Auto')
tokenizer = AutoTokenizer.from_pretrained(base_model_path)

# If model is LORA, load LORA model
if is_lora:
  # Check if directory has adapter config and model files
  for file in filelist:
    if file.endswith(('adapter_config.json', 'adapter_model.bin')):
      print("{} found, Valid LORA model".format(file))
  lora_model_path = model_path
  lora_model = PeftModel.from_pretrained(base_model, lora_model_path)
  model = lora_model
  print("=========================================================\
        Successfully loaded LORA Model: {}".format(model_name))
# Else load base model
else:
  model = base_model
  print("=========================================================\
        Successfully loaded Base Model: {}".format(model_name))
template = """Question: {question}

Answer:\n"""
prompt = PromptTemplate(template=template, input_variables=["question"])
print(prompt)

# max_length has typically been deprecated for max_new_tokens 
max_tokens = 64
pipe = pipeline(
    "text-generation", model=model, tokenizer = tokenizer,
      max_new_tokens=max_tokens, model_kwargs={"temperature":0})
hf = HuggingFacePipeline(pipeline=pipe)

llm_chain = LLMChain(prompt=prompt, llm=hf)
question = args.question
print(llm_chain.run(question))