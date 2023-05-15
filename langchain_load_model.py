# Script for loading a local model to interact with langchain
from langchain import PromptTemplate, LLMChain
from langchain.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


template = """Question: {question}

Answer:"""
prompt = PromptTemplate(template=template, input_variables=["question"])
print(prompt)

# Model path (directory folder) contains pytorch.bin/flaxweights/model.safetensors and tokenizer

model_path = "facebook_galactica-125m"
model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# max_length has typically been deprecated for max_new_tokens 
pipe = pipeline(
    "text-generation", model=model, tokenizer = tokenizer,
      max_new_tokens=64, model_kwargs={"temperature":0})
hf = HuggingFacePipeline(pipeline=pipe)

llm_chain = LLMChain(prompt=prompt, llm=hf)
question = "What is life?"
print(llm_chain.run(question))