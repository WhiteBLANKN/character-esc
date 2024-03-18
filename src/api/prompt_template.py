from string import Template

def process_prompt(character: str) -> str:
    
    default_prompt_template = Template(
    "请把下方对话的assistant部分改写为$character，并保持$character的语气。"
    "注意请不要修改对话内涵，只需要融入人物风格特征。"
    """返回多轮对话格式如下:
    {'role': $character, 'content': content},
    {'role': user, 'content': content}，
    ...
    {'role': $character, 'content': content},
    {'role': user, 'content': content}
    """
    )
    
    prompt = default_prompt_template.substitute(
        {'character':character}
    )
    
    return prompt