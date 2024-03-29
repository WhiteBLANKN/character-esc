import json
from datasets import Dataset

data_path = "/home/shijiajie/github/character-esc/dataset/ExTES.json"
with open(data_path, 'r', encoding='utf-8') as files:
    json_data = json.loads(files.read())
    
def check(i):
    for k, v in i.items():
        if not k in ['scene', 'description', 'content']:
            return False
    return True

cleaned_json_dict = [i for i in json_data if check(i)]

_STRATEGY = (
    'Reflective Statements',
    'Clarification',
    'Emotional Validation',
    'Empathetic Statements',
    'Affirmation',
    'Offer Hope',
    'Avoid Judgment And Criticism',
    'Suggest Options',
    'Collaborative Planning',
    'Provide Different Perspectives',
    'Reframe Negative Thoughts',
    'Share Information',
    'Normalize Experiences',
    'Promote Self-Care Practices',
    'Stress Management',
    'Others'
)

def extes_template(example):
    conversation = ''
    for conversation_dicts in example['content']:
        for k, v in conversation_dicts.items():
            if k == 'User':
                conversation += ("<|im_start|>user" + "\n" + v + "<|im_end|>\n")
            elif k == 'AI':
                strategy = conversation_dicts['AI Strategy']
                if strategy not in _STRATEGY:
                    conversation += ("<|im_start|>assistant" + '(`Others`)' + "\n" + v + "<|im_end|>\n")
                else:
                    conversation += ("<|im_start|>assistant" + f'(`{strategy}`)' + "\n" + v + "<|im_end|>\n")
    return conversation

def get_dataset():
    scene, description, content = [], [], []
    index = -1
    for example in cleaned_json_dict:
        index += 1
        try:
            _scene = example['scene']
            _description = example['description']
            _content = extes_template(example)
            assert type(_scene) == str and type(_description) == str and type(_content)
        except:
            print(f"Can not process Data NO.{index:<5}.And it had been Skipped.")
            continue
        scene.append(_scene)
        description.append(_description)
        content.append(_content)
        
    dataset_dict = {
    'scene': scene,
    'description': description,
    'content': content
    }
    
    raw_datasets = Dataset.from_dict(dataset_dict)
    
    return raw_datasets