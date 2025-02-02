# import openai
from openai import OpenAI
import sys
import time
import json

DYANAMIC_KGQA_PATH = '../data/dynamickgqa_test.json'

def run_llm(prompt, temperature, max_tokens, opeani_api_keys, engine="gpt-3.5-turbo"):
    if "llama" in engine.lower():
        sys.exit("Llama is not supported in this version.")  
        # openai.api_key = "EMPTY"
        # openai.api_base = "http://localhost:8000/v1"  # your local llama server port
        # engine = openai.Model.list()["data"][0]["id"]
    else:
        # This operation of creating a client on each call is not efficient.
        # This will be fixed in the next version.
        client = OpenAI(api_key=opeani_api_keys)
        # openai.api_key = opeani_api_keys

    messages = [{"role":"system","content":"You are an AI assistant that helps people find information."}]
    message_prompt = {"role":"user","content":prompt}
    messages.append(message_prompt)
    # print("start openai")
    f = 0
    while(f == 0):
        try:
            response = client.chat.completions.create(
                    model=engine,
                    messages = messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    frequency_penalty=0,
                    presence_penalty=0)
            result = response.choices[0].message.content
            f = 1
        except:
            print("openai error, retry")
            time.sleep(2)
    # print("end openai")
    return result

def prepare_dataset(dataset_name):
    if dataset_name == 'cwq':
        with open('../data/cwq.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'question'
    elif dataset_name == 'webqsp':
        with open('../data/WebQSP.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'RawQuestion'
    elif dataset_name == 'grailqa':
        with open('../data/grailqa.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'question'
    elif dataset_name == 'simpleqa':
        with open('../data/SimpleQA.json',encoding='utf-8') as f:
            datas = json.load(f)    
        question_string = 'question'
    elif dataset_name == 'qald':
        with open('../data/qald_10-en.json',encoding='utf-8') as f:
            datas = json.load(f) 
        question_string = 'question'   
    elif dataset_name == 'webquestions':
        with open('../data/WebQuestions.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'question'
    elif dataset_name == 'trex':
        with open('../data/T-REX.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'input'    
    elif dataset_name == 'zeroshotre':
        with open('../data/Zero_Shot_RE.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'input'    
    elif dataset_name == 'creak':
        with open('../data/creak.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'sentence'
    elif dataset_name == 'dynamickgqa':
        with open(DYANAMIC_KGQA_PATH,encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'question'
    else:
        print("dataset not found")
        exit(-1)
    return datas, question_string