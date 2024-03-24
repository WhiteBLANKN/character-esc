from openai import OpenAI
import argparse
from utils import process_prompt, clean_screen
import os

parser = argparse.ArgumentParser()
parser.add_argument("--api_key", help="Your API key from gpt4 or moonshot.", required=True)
parser.add_argument("--use_moonshot", action='store_true', help="From OPENAI or MOONSHOT", required=True)
parser.add_argument("--user_prompt", help="The prompt to be sent to LLM.")
parser.add_argument("--system_prompt", help="The system prompt will be set to LLM.")
parser.add_argument("--character", help="The character to be trained.", required=True)
parser.add_argument("--model_version", help="Which version of model you select to use. Make sure it's consistent with your api provider.", required=True)
parser.add_argument("--temperature", help="This parameter defines the varities of Model response.", default=0.3)

args = parser.parse_args()

api_key = os.getenv(args.api_key)
use_moonshot = args.use_moonshot
character = args.character
model_version = args.model_version
temperature= args.temperature

def main():

    #检查是否传入自定义的prompt
    if args.user_prompt is not None:
        print("You use custom Prompt as below:\n")
        print(f"{args.prompt}")
        user_prompt = args.prompt
    else:
        user_prompt = process_prompt(character)
        
    #检查是否使用了Moonshot的API
    if args.use_moonshot:
        base_url="https://api.moonshot.cn/v1"
        if args.system_prompt is not None:
            system_prompt = args.system_prompt
        else:
            system_prompt = (
                "你是 Kimi，由 Moonshot AI 提供的人工智能助手，你更擅长中文和英文的对话。"
                "你会为用户提供安全，有帮助，准确的回答。"
                "同时，你会拒绝一切涉及恐怖主义，种族歧视，黄色暴力等问题的回答。Moonshot AI 为专有名词，不可翻译成其他语言。"
                )
    else:
        #调用OPENAI的api确保能否科学上网
        base_url=None
        if args.system_prompt is not None:
            system_prompt = args.system_prompt
        else:
            system_prompt = (
                "You are a helpful assistant."
                )
        

    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
    )
    #请求API的响应
    completion = client.chat.completions.create(
        model=model_version,
        messages=[ 
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=temperature,
    )
    #清理屏幕并打印当前生成中的消息
    clean_screen()
    print(completion.choices[0].message)

if __name__ == "__main__":
    main()