import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings
warnings.filterwarnings('ignore')

peft_model_id="/root/models/llm/bloom-1b1-lora" # lora adapter weights location
base_model_path="/root/models/llm/bloom-1b1" # base model location
​
model = AutoModelForCausalLM.from_pretrained(base_model_path, return_dict=True,  device_map='auto')
tokenizer = AutoTokenizer.from_pretrained(base_model_path)
model = PeftModel.from_pretrained(model, peft_model_id)
​
def get_model_response(prompt_input,max_new_tokens_=50):
    full_prompt="""Below is an instruction that describes a task. Write a response that appropriately completes 
    the request. ### Instruction: {} ### Response:""".format(prompt_input)
    batch = tokenizer(full_prompt, return_tensors='pt')
    with torch.cuda.amp.autocast():
        output_tokens = model.generate(**batch, max_new_tokens=max_new_tokens_)
    response=tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    response=response.removeprefix(full_prompt).strip().removeprefix('->: ')
    print('\n', response,'\n')