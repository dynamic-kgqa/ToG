from tqdm import tqdm
import argparse
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial

from utils import *
from yago_func import *
import random
# from client import *
import time

def process(data, args, question_string) -> tuple[str, str, list]:
    question = data[question_string]
    results = ""
    try:
        topic_entity = data['qid_topic_entity']
        cluster_chain_of_entities = []
        if len(topic_entity) == 0:
            results = generate_without_explored_paths(question, args)
            # save_2_jsonl(question, results, [], file_name=args.dataset)
            return question, results, []
        pre_relations = []
        pre_heads= [-1] * len(topic_entity)
        flag_printed = False
        for depth in range(1, args.depth+1):
            current_entity_relations_list = []
            i=0
            for entity in topic_entity:
                if entity!="[FINISH_ID]":
                    retrieve_relations_with_scores = relation_search_prune(entity, topic_entity[entity], pre_relations, pre_heads[i], question, args)  # best entity triplet, entitiy_id
                    current_entity_relations_list.extend(retrieve_relations_with_scores)
                i+=1
            total_candidates = []
            total_scores = []
            total_relations = []
            total_entities_id = []
            total_topic_entities = []
            total_head = []

            for entity in current_entity_relations_list:
                if entity['head']:
                    entity_candidates_id = entity_search(entity['entity'], entity['relation'], True)
                else:
                    entity_candidates_id = entity_search(entity['entity'], entity['relation'], False)
                
                if args.prune_tools == "llm":
                    if len(entity_candidates_id) >=20:
                        entity_candidates_id = random.sample(entity_candidates_id, args.num_retain_entity)

                if len(entity_candidates_id) ==0:
                    continue
                scores, entity_candidates, entity_candidates_id = entity_score(question, entity_candidates_id, entity['score'], entity['relation'], args)
                
                total_candidates, total_scores, total_relations, total_entities_id, total_topic_entities, total_head = update_history(entity_candidates, entity, scores, entity_candidates_id, total_candidates, total_scores, total_relations, total_entities_id, total_topic_entities, total_head)
            
            if len(total_candidates) ==0:
                results = half_stop_no_write(question, cluster_chain_of_entities, depth, args)
                flag_printed = True
                break
                
            flag, chain_of_entities, entities_id, pre_relations, pre_heads = entity_prune(total_entities_id, total_relations, total_candidates, total_topic_entities, total_head, total_scores, args)
            cluster_chain_of_entities.append(chain_of_entities)
            if flag:
                stop, results = reasoning(question, cluster_chain_of_entities, args)
                if stop:
                    print("ToG stoped at depth %d." % depth)
                    # save_2_jsonl(question, results, cluster_chain_of_entities, file_name=args.dataset)
                    flag_printed = True
                    break
                else:
                    print("depth %d still not find the answer." % depth)
                    flag_finish, entities_id = if_finish_list(entities_id)
                    if flag_finish:
                        results = half_stop_no_write(question, cluster_chain_of_entities, depth, args)
                        flag_printed = True
                    else:
                        topic_entity = {entity: id2entity_name_or_type(entity) for entity in entities_id}
                        continue
            else:
                results = half_stop_no_write(question, cluster_chain_of_entities, depth, args)
                flag_printed = True
        
        if not flag_printed:
            results = generate_without_explored_paths(question, args)
            # save_2_jsonl(question, results, [], file_name=args.dataset)
            return question, results, []
        return question, results, cluster_chain_of_entities
    except Exception as e:
        results = generate_without_explored_paths(question, args)
        results = "Error, falling back to LLM: " + (results if results is not None else "null")
        # save_2_jsonl(question, results, [], file_name=args.dataset)
        return question, results, []

def main(args):
    datas, question_string = prepare_dataset(args.dataset)
    print("Start Running ToG on %s dataset." % args.dataset)
    start, end = args.start, args.end

    datas = datas[start:end]

    if args.avoid_existing:
        json_backup_path = get_jsonl_path_for_backup(args.dataset)
        with open(json_backup_path, "r") as json_backup_ref:
            existing_answers = [json.loads(line) for line in json_backup_ref]
        datas = avoid_existing(datas, existing_answers, question_string)
    print("Number of existing questions: ", len(existing_answers))
    print("Number of questions: ", len(datas))

    batch_size = args.batch_size
    for i in range(0, len(datas), batch_size):
        print("Processing batch %d" % (i//batch_size))
        output = main_batch(datas[i:i+batch_size], question_string, args)
        save_2_jsonl_batch(output, file_name=args.dataset)

    print("Finish Running ToG on %s dataset." % args.dataset)

def main_batch(datas, question_string, args):
    start_time = time.time()
    output = []

    with ThreadPoolExecutor(max_workers=args.n) as executor:
        futures = [
            executor.submit(
                partial(
                    process, args=args, question_string=question_string
                ), data
            )
            for data in datas
        ]
        for future in tqdm(as_completed(futures), total=len(datas)):
            try:
                question, result, cluster_chain_of_entities = future.result(timeout=1)
                # save_2_jsonl(question, result, cluster_chain_of_entities, file_name=args.dataset)
                output_dict = {"question":question, "results": result, "reasoning_chains": cluster_chain_of_entities}
                output.append(output_dict)
            except Exception as e:
                print("Timeout Error:", str(e)[:250])
    
    print("time taken in minutes", (time.time()-start_time)/60)
    
    return output

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str,
                        default="webqsp", help="choose the dataset.")
    parser.add_argument("--max_length", type=int,
                        default=256, help="the max length of LLMs output.")
    parser.add_argument("--temperature_exploration", type=float,
                        default=0.4, help="the temperature in exploration stage.")
    parser.add_argument("--temperature_reasoning", type=float,
                        default=0, help="the temperature in reasoning stage.")
    parser.add_argument("--width", type=int,
                        default=3, help="choose the search width of ToG.")
    parser.add_argument("--depth", type=int,
                        default=3, help="choose the search depth of ToG.")
    parser.add_argument("--remove_unnecessary_rel", type=bool,
                        default=True, help="whether removing unnecessary relations.")
    parser.add_argument("--LLM_type", type=str,
                        default="gpt-3.5-turbo", help="base LLM model.")
    parser.add_argument("--opeani_api_keys", type=str,
                        default="", help="if the LLM_type is gpt-3.5-turbo or gpt-4, you need add your own openai api keys.")
    parser.add_argument("--num_retain_entity", type=int,
                        default=5, help="Number of entities retained during entities search.")
    parser.add_argument("--prune_tools", type=str,
                        default="llm", help="prune tools for ToG, can be llm (same as LLM_type), bm25 or sentencebert.")
    parser.add_argument("--start", type=int,
                        default=0, help="start index.")
    parser.add_argument("--end", type=int,
                        default=-1, help="end index.")
    parser.add_argument("--n", type=int,
                        default=2, help="number of processes.")
    parser.add_argument("--batch_size", type=int,
                        default=500, help="batch size.")
    parser.add_argument("--avoid_existing", type=bool,
                        default=True, help="avoid reprocessing existing rows.")
    args = parser.parse_args()

    main(args)


# Command
# python main_yago.py --dataset webqsp --max_length 256 --temperature_exploration 0.4 --temperature_reasoning 0 --width 3 --depth 3 --remove_unnecessary_rel True --LLM_type mistral --num_retain_entity 5 --prune_tools llm --n 1
# python main_yago.py --dataset cwq --max_length 256 --temperature_exploration 0.4 --temperature_reasoning 0 --width 3 --depth 3 --remove_unnecessary_rel True --LLM_type anthropic --num_retain_entity 5 --prune_tools llm --n 1;
# python main_yago.py --dataset grailqa --max_length 256 --temperature_exploration 0.4 --temperature_reasoning 0 --width 3 --depth 3 --remove_unnecessary_rel True --LLM_type anthropic --num_retain_entity 5 --prune_tools llm --n 1;
# python main_yago.py --dataset qald --max_length 256 --temperature_exploration 0.4 --temperature_reasoning 0 --width 3 --depth 3 --remove_unnecessary_rel True --LLM_type anthropic --num_retain_entity 5 --prune_tools llm --n 1;
# python main_yago.py --dataset simpleqa --max_length 256 --temperature_exploration 0.4 --temperature_reasoning 0 --width 3 --depth 3 --remove_unnecessary_rel True --LLM_type anthropic --num_retain_entity 5 --prune_tools llm --n 1;
# python main_yago.py --dataset webqsp --max_length 256 --temperature_exploration 0.4 --temperature_reasoning 0 --width 3 --depth 3 --remove_unnecessary_rel True --LLM_type anthropic --num_retain_entity 5 --prune_tools llm --n 1;