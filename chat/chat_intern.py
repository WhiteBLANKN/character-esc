import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import os
from modelscope import snapshot_download
import sys

#供大家测试，后续将删除
if sys.argv[1] == 'download_from_modelscope':
    model_path = snapshot_download('Shanghai_AI_Laboratory/internlm2-chat-7b')
else:
    model_path = sys.argv[1]

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True
)
max_turns = 1024


def chat():
    length = 0
    model = AutoModelForCausalLM.from_pretrained(model_path, quantization_config=bnb_config, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = model.eval()
    history=[]
    os.system('clear')
    for turn in range(max_turns):
        query = input("User: ")
        print('\n' + "InternLM: ", end="")
        for response, history in model.stream_chat(tokenizer, query, history):
            print(response[length:], flush=True, end="")
            length = len(response)
        print('\n')
        chat_pair = (query, response)
        history.append(chat_pair)

if __name__ == '__main__':
    chat()
