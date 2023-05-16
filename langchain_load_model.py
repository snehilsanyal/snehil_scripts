# Script for loading a local model to interact with langchain
import argparse 
import os 
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel 
from langchain import PromptTemplate, LLMChain
from langchain.llms import HuggingFacePipeline

parser = argparse.ArgumentParser(description = 'Load locally hosted model from server\
                                 for further usage via LangChain')
parser.add_argument('-m', '--model_name')
args = parser.parse_args()

template = """Question: {question}

Answer:\n"""
prompt = PromptTemplate(template=template, input_variables=["question"])
print(prompt)

# Model path (directory folder) contains pytorch.bin/flaxweights/model.safetensors and tokenizer
llm_dir = '/root/models/llm/'
llm_model_names = os.listdir(llm_dir)
# Return models related to bloom-1b1 (Later for further automation)
# [model_name for model_name in model_names if 'bloom-1b1' in model_name]
base_model_path = llm_dir + args.model_name
is_base = True 

## PEFT tuners
# Check whether model is LORA model
is_lora = is_base and ('lora' in args.model_name)
filelist = os.listdir(base_model_path)
for file in filelist:
    if is_lora and file.endswith(('adapter_config.json', 'adapter_model.bin')):
        is_lora = True 

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(base_model_path, return_dict = True, device_map = 'Auto')
tokenizer = AutoTokenizer.from_pretrained(base_model_path)

# If model is LORA load peft model
if is_lora:
    peft_model_path = base_model_path
    peft_model = PeftModel.from_pretrained(base_model, peft_model_id)
    model = peft_model
    print("=========================================================\
          Successfully loaded LORA Model: {}".format(args.model_name))
# Else load base model
else:
    model = base_model
    print("=========================================================\
          Successfully loaded Base Model: {}".format(args.model_name))

# max_length has typically been deprecated for max_new_tokens 
pipe = pipeline(
    "text-generation", model=model, tokenizer = tokenizer,
      max_new_tokens=32, model_kwargs={"temperature":0})
hf = HuggingFacePipeline(pipeline=pipe)

llm_chain = LLMChain(prompt=prompt, llm=hf)
question = "What is life?"
print(llm_chain.run(question))