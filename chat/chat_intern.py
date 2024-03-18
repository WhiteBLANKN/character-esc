import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import os

model_path = "/home/shijiajie/github/internlm2-chat-7b"
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