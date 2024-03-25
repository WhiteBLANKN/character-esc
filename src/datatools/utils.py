import json
import re
import os


def post_process_response(result):
    
    pattern = re.compile(r'\{[^}]+\}', re.DOTALL)
    matches = pattern.findall(result)
    dialogs = [json.loads(json_str) for json_str in matches]
    
    return dialogs

def check_print(dialogs):
    # windows清屏指令
    if os.name == 'nt':
        os.system('cls')
    # macos或者linux清屏指令
    else:
        os.system('clear')
        
    for dialog in dialogs:
        role = dialog['role']
        content = dialog['content']
        print(f'{role}: {content}\n')