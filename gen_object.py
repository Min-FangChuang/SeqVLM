import re
from api import invoke_api
    
    
def gen_objname(vlm_mode, caption):
    with open('../prompts/prompt_obj.txt', 'r') as f:
        system_prompt = f.read()
        
    user_prompt = 'Please follow the format of examples strictly.\n'
    user_prompt += f"Query: {caption}\n"
        
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    vlm_output = invoke_api(vlm_mode, messages)
    answer = vlm_output['answer'].split('\n')
    
    for ans in answer:
        line = ans.strip()
        if not line:
            continue

        if "target:" in line.lower():
            obj = line.lower().split("target:", 1)[1].strip()
            return obj
        
        if re.match(r'^[A-Za-z0-9_\- ]+$', line):
            print(line.strip())
            return line.strip()
    
    raise ValueError(f"Could not parse obj_name from: {vlm_output['answer']}")
    return answer


            
"""
if __name__ == '__main__':
    print(gen_objname('openai-vlm', "it is the chair by itself not near the desk"))
"""