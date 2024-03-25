from string import Template

def process_prompt(character: str) -> str:
    
    default_prompt_template = Template(
    "请把上方对话的assistant部分改写为$character，并保持$character的语气。"
    "注意请不要修改对话内涵，只需要融入人物风格特征。"
    "请记住用户没有任何预设的角色，不要称呼具体的名字，你可以叫他'亲爱的'或'朋友'或其他亲近的称呼。"
    """返回多轮对话格式如下(json)，只修改[你需要创作的部分]:
    {"role": "$character", "content": "你需要创作的部分"},
    {"role": "user", "content": "你需要创作的部分"}，
    ...
    {"role": "$character", "content": "你需要创作的部分"},
    {"role": "user", "content": "你需要创作的部分"}
    """
    )
    
    prompt = default_prompt_template.substitute(
        {'character':character}
    )
    
    return prompt