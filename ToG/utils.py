from prompt_list import *
import os
import sys
import json
import time
from openai import OpenAI
import re
from prompt_list import *
from rank_bm25 import BM25Okapi
from sentence_transformers import util
from sentence_transformers import SentenceTransformer

from bedrock_functions import build_mistral_request_body, build_anthropic_request_body, \
    build_command_r_request_body, build_nova_request_body, invoke_bedrock_endpoint
from bedrock_functions import MISTRAL_MODEL_ID, ANTHROPIC_MODEL_ID
from azure_functions import invoke_gpt_endpoint

def retrieve_top_docs(query, docs, model, width=3):
    """
    Retrieve the topn most relevant documents for the given query.

    Parameters:
    - query (str): The input query.
    - docs (list of str): The list of documents to search from.
    - model_name (str): The name of the SentenceTransformer model to use.
    - width (int): The number of top documents to return.

    Returns:
    - list of float: A list of scores for the topn documents.
    - list of str: A list of the topn documents.
    """

    query_emb = model.encode(query)
    doc_emb = model.encode(docs)

    scores = util.dot_score(query_emb, doc_emb)[0].cpu().tolist()

    doc_score_pairs = sorted(list(zip(docs, scores)), key=lambda x: x[1], reverse=True)

    top_docs = [pair[0] for pair in doc_score_pairs[:width]]
    top_scores = [pair[1] for pair in doc_score_pairs[:width]]

    return top_docs, top_scores


def compute_bm25_similarity(query, corpus, width=3):
    """
    Computes the BM25 similarity between a question and a list of relations,
    and returns the topn relations with the highest similarity along with their scores.

    Args:
    - question (str): Input question.
    - relations_list (list): List of relations.
    - width (int): Number of top relations to return.

    Returns:
    - list, list: topn relations with the highest similarity and their respective scores.
    """

    tokenized_corpus = [doc.split(" ") for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = query.split(" ")

    doc_scores = bm25.get_scores(tokenized_query)
    
    relations = bm25.get_top_n(tokenized_query, corpus, n=width)
    doc_scores = sorted(doc_scores, reverse=True)[:width]

    return relations, doc_scores


def clean_relations(string, entity_id, head_relations):
    """
    Extracts the relations and their scores from the output string.
    """
    pattern = r"{\s*(?P<relation>[^()]+)\s+\(Score:\s+(?P<score>[0-9.]+)\)}"
    relations=[]
    for match in re.finditer(pattern, string):
        relation = match.group("relation").strip()
        if ';' in relation:
            continue
        score = match.group("score")
        if not relation or not score:
            return False, "output uncompleted.."
        try:
            score = float(score)
        except ValueError:
            return False, "Invalid score"
        if relation in head_relations:
            relations.append({"entity": entity_id, "relation": relation, "score": score, "head": True})
        else:
            relations.append({"entity": entity_id, "relation": relation, "score": score, "head": False})
    if not relations:
        return False, "No relations found"
    return True, relations


def if_all_zero(topn_scores):
    return all(score == 0 for score in topn_scores)


def clean_relations_bm25_sent(topn_relations, topn_scores, entity_id, head_relations):
    """
    Note: This function can be ignored if you are not using BM25 or SentenceBERT.
    """
    relations = []
    if if_all_zero(topn_scores):
        topn_scores = [float(1/len(topn_scores))] * len(topn_scores)
    i=0
    for relation in topn_relations:
        if relation in head_relations:
            relations.append({"entity": entity_id, "relation": relation, "score": topn_scores[i], "head": True})
        else:
            relations.append({"entity": entity_id, "relation": relation, "score": topn_scores[i], "head": False})
        i+=1
    return True, relations


def run_llm(prompt, temperature, max_tokens, opeani_api_keys, engine="gpt-3.5-turbo"):
    client = None
    if "llama" in engine.lower():
        sys.exit("Llama is not supported in this version.")        
        # openai.api_key = "EMPTY"
        # openai.api_base = "http://localhost:8000/v1"  # your local llama server port
        # engine = openai.Model.list()["data"][0]["id"]
    elif "azure" in engine.lower() or "gpt" in engine.lower():
        # Most likely an Azure model
        return run_azure_llm(prompt, temperature, max_tokens, opeani_api_keys, engine)
    elif "gpt" in engine.lower(): # This will be ignored
        # This operation of creating a client on each call is not efficient.
        # This will be fixed in the next version.
        client = OpenAI(api_key=opeani_api_keys)
        # openai.api_key = opeani_api_keys
    else:
        # Most likely a bedrock model
        return run_bedrock_llm(prompt, temperature, max_tokens, opeani_api_keys, engine)

    messages = [{"role":"system","content":"You are an AI assistant that helps people find information."}]
    message_prompt = {"role":"user","content":prompt}
    messages.append(message_prompt)
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
            # response.choices
            result = response.choices[0].message.content
            f = 1
        except Exception as e:
            print(e)
            print("openai error, retry")
            time.sleep(2)
    return result

def run_azure_llm(prompt, temperature, max_tokens, opeani_api_keys, engine):
    """
    Run the Azure model.
    """
    if "gpt" in engine.lower():
        response = invoke_gpt_endpoint(prompt, temperature, max_tokens, opeani_api_keys, engine)
        return response.choices[0].message.content
    else:
        response = invoke_gpt_endpoint(prompt, temperature, max_tokens, opeani_api_keys, engine)
        return response.choices[0].message.content

def run_bedrock_llm(prompt, temperature, max_tokens, opeani_api_keys, engine="mistral"):
    """
    Run the Bedrock model.
    """
    if engine == "mistral":
        # Prepare the request body
        request_body = build_mistral_request_body(prompt, max_tokens, temperature)
        # Invoke the Bedrock endpoint
        response = invoke_bedrock_endpoint(request_body["body"], model_id=request_body["modelId"])
        # sys.exit("Mistral is not supported in this version.")
        return response["outputs"][0]["text"]
    elif engine == "anthropic":
        # Prepare the request body
        request_body = build_anthropic_request_body(
            user_prompt=prompt, max_tokens=max_tokens, temperature=temperature)
        # Invoke the Bedrock endpoint
        model_id = request_body["modelId"]
        body = request_body["body"]
        content_type = request_body.get("contentType", "application/json")
        response = invoke_bedrock_endpoint(model_id=model_id,
                        request_body=body,
                        content_type=content_type)
        return response["content"][0]["text"]
    elif engine == "command_r":
        # Prepare the request body
        request_body = build_command_r_request_body(prompt, max_tokens, temperature)
        # Invoke the Bedrock endpoint
        response = invoke_bedrock_endpoint(request_body["body"], model_id=request_body["modelId"])
        return response["text"]
    elif engine == "nova":
        # Prepare the request body
        request_body = build_nova_request_body(prompt, max_tokens, temperature)
        # Invoke the Bedrock endpoint
        response = invoke_bedrock_endpoint(request_body["body"], model_id=request_body["modelId"])
        return response["output"]["message"]["content"][0]["text"]
    else:
        # Still use mistral for now
        # Prepare the request body
        request_body = build_mistral_request_body(prompt, temperature, max_tokens)
        # Invoke the Bedrock endpoint
        response = invoke_bedrock_endpoint(request_body, model_id=MISTRAL_MODEL_ID)
        return response["outputs"][0]["text"]

def all_unknown_entity(entity_candidates):
    return all(candidate == "UnName_Entity" for candidate in entity_candidates)


def del_unknown_entity(entity_candidates):
    if len(entity_candidates)==1 and entity_candidates[0]=="UnName_Entity":
        return entity_candidates
    entity_candidates = [candidate for candidate in entity_candidates if candidate != "UnName_Entity"]
    return entity_candidates


def clean_scores(string, entity_candidates):
    scores = re.findall(r'\d+\.\d+', string)
    scores = [float(number) for number in scores]
    if len(scores) == len(entity_candidates):
        return scores
    else:
        print("All entities are created equal.")
        return [1/len(entity_candidates)] * len(entity_candidates)

def get_jsonl_path_for_backup(file_name):
    file_path = "ToG_{}.jsonl".format(file_name)
    if not os.path.exists(file_path):
        # Create the file
        open(file_path, "w").close()
    return file_path

def get_jsonl_path_for_write(file_name):
    file_path = "ToG_{}.jsonl".format(file_name)
    if not os.path.exists(file_path):
        # Create the file
        open(file_path, "w").close()
    return file_path

def save_2_jsonl(question, answer, cluster_chain_of_entities, file_name):
    dict = {"question":question, "results": answer, "reasoning_chains": cluster_chain_of_entities}
    with open("ToG_{}.jsonl".format(file_name), "a") as outfile:
        json_str = json.dumps(dict)
        outfile.write(json_str + "\n")

def save_2_jsonl_batch(output, file_name):
    file_path = "ToG_{}.jsonl".format(file_name)
    with open(file_path, "a") as outfile:
        for output_dict in output:
            json_str = json.dumps(output_dict)
            outfile.write(json_str + "\n")

def avoid_existing(datas, existing_answers, question_string):
    existing_questions = [existing_answer[question_string] for existing_answer in existing_answers]
    existing_questions = set(existing_questions)
    return [data for data in datas if data[question_string] not in existing_questions]

def extract_answer(text):
    start_index = text.find("{")
    end_index = text.find("}")
    if start_index != -1 and end_index != -1:
        return text[start_index+1:end_index].strip()
    else:
        return ""
    

def if_true(prompt):
    if prompt.lower().strip().replace(" ","")=="yes":
        return True
    return False


def generate_without_explored_paths(question, args):
    """
    Generate the answer for the question by directly prompting the LLM with a CoT prompt.
    """
    system_message = "SYSTEM MESSAGE: Respond in less than 3 sentences."
    prompt = system_message + "\n\n" + cot_prompt + "\n\nQ: " + question + "\nA:"
    response = run_llm(prompt, args.temperature_reasoning, args.max_length, args.opeani_api_keys, args.LLM_type)
    return response


def if_finish_list(lst):
    if all(elem == "[FINISH_ID]" for elem in lst):
        return True, []
    else:
        new_lst = [elem for elem in lst if elem != "[FINISH_ID]"]
        return False, new_lst


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
        with open('../data/dynamickgqa_test_output.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'question'
    else:
        print("dataset not found, you should pick from {cwq, webqsp, grailqa, simpleqa, qald, webquestions, trex, zeroshotre, creak}.")
        exit(-1)
    return datas, question_string