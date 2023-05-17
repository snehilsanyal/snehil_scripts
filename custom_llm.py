## Script to create Custom LLM Class 
# LangChain Issue, GitHub: https://github.com/hwchase17/langchain/issues/737
# 1. Try out a structure in the repo in form of a library (so that everything can be imported)
# 2. Create custom LLM using a locally hosted model and tokenizer
# 3. Merge 2 LLMs into a single pipeline/CustomLLM/ using this script
import os, argparse
import torch
from langchain import PromptTemplate, LLMChain
# base.py from LLM on top of which we will build custom LLM
from langchain.llms.base import LLM
from transformers import AutoModelForCausalML, AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

class CustomLLM(LLM):
    model_name = ''
    tokenizer = ''
    pipe = pipeline(task = 'text2text-generation', model = model_name,
                        device = 'cuda:0', tokenizer = tokenizer, 
                        model_kwargs = {"torch_dtype":torch.bfloat16}) 

    def _call(self, prompt, stop=None):
        return self.pipeline(prompt, max_length=9999)[0]["generated_text"]

    def _identifying_params(self):
        return {"name_of_model": self.model_name}

    def _llm_type(self):
        return "custom"

prompt = """ """
llm = CustomLLM(temperature = 0)
llm_chain = LLMChain(prompt = prompt, llm = llm)