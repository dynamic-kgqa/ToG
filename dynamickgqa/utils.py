from openai import OpenAI
import sys
import time
import json
import spacy

DYANAMIC_KGQA_PATH = '../data/dynamickgqa_test.json'

def run_llm(prompt, temperature, max_tokens, opeani_api_keys, engine="gpt-4.0"):
    if "llama" in engine.lower():
        sys.exit("Llama is not supported in this version.")  
    else:
        client = OpenAI(api_key=opeani_api_keys)

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

def prepare_dataset(dataset_path):
    with open(dataset_path,encoding='utf-8') as f:
        datas = json.load(f)
    question_string = 'question'
    return datas, question_string

def get_entities(text):
    # Break into lines, remove empty lines and those that do not start with a number
    lines = text.split("\n")
    lines = [line for line in lines if line.strip() != "" and line.strip()[0].isdigit()]
    # Extract entities on each line which comes after the first number
    entities = [line.split(" ", 1)[1] for line in lines]
    return entities

def setup_ner():
    nlp = spacy.load("en_core_web_sm")
    # nlp.add_pipe("entityLinker", last=True)
    return nlp

def get_spacy_entities(nlp, text):
    doc = nlp(text)
    return doc.ents