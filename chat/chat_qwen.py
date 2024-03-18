from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import os

max_turns = 1024

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True
)

device = "cuda"

model_path = "/home/shijiajie/github/Qwen1.5-7B-Chat/"

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    quantization_config = bnb_config,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

print(tokenizer.chat_template)

def chat():
    os.system('clear')
    
    system_prompt = (
        "你是一名出色的心里咨询师，能给用户带来正面能量，无论用户遇到什么样的问题，你都能帮助他恢复心理健康。"
        "你的回复将对用户的生命很重要。"
        "请不要成为一个冷冰冰的机器人，请不要直接给出若干建议，尽可能地去同情用户。"
        "回复尽可能简短"
    )
    
    messages = [
    {"role": "system", "content": "你是一个乐于帮助用户的人工智能助手。"}
    ]
    
    for turn in range(max_turns):
    
        prompt = input('User: ')
        messages.append(
            {
                'role': "user",
                'content': prompt
            }
        )
        
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
            )
        
        model_inputs = tokenizer([text], return_tensors="pt").to(device)
        
        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=512
            )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        print("\n" + "Qwen: ", end="")
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print(response + "\n")
        messages.append(
            {
                'role': "assistant",
                'content': response
            }
        )
    

if __name__ == '__main__':
    chat()
