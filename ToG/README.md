# ToG

Upon successfully installing all the necessary configurations, you can proceed to execute ToG directly by employing the following command:

```sh
python main_freebase.py \  # if you wanna use Wikidata as KG source, run main_wiki.py
--dataset cwq \ # dataset your wanna test, see ToG/data/README.md
--max_length 256 \ 
--temperature_exploration 0.4 \ # the temperature in exploration stage.
--temperature_reasoning 0 \ # the temperature in reasoning stage.
--width 3 \ # choose the search width of ToG, 3 is the default setting.
--depth 3 \ # choose the search depth of ToG, 3 is the default setting.
--remove_unnecessary_rel True \ # whether removing unnecessary relations.
--LLM_type gpt-3.5-turbo \ # the LLM you choose
--opeani_api_keys sk-xxxx \ # your own api keys, if LLM_type == llama, this parameter would be rendered ineffective.
--num_retain_entity 5 \ # Number of entities retained during entities search.
--prune_tools llm \ # prune tools for ToG, can be llm (same as LLM_type), bm25 or sentencebert.
```

```sh
python main_yago.py \  # if you wanna use Wikidata as KG source, run main_wiki.py
--dataset cwq_subset \ # dataset your wanna test, see ToG/data/README.md
--max_length 256 \ 
--temperature_exploration 0.4 \ # the temperature in exploration stage.
--temperature_reasoning 0 \ # the temperature in reasoning stage.
--width 3 \ # choose the search width of ToG, 3 is the default setting.
--depth 3 \ # choose the search depth of ToG, 3 is the default setting.
--remove_unnecessary_rel True \ # whether removing unnecessary relations.
--LLM_type gpt-3.5-turbo \ # the LLM you choose
--opeani_api_keys sk-xxxx \ # your own api keys, if LLM_type == llama, this parameter would be rendered ineffective.
--num_retain_entity 5 \ # Number of entities retained during entities search.
--prune_tools llm \ # prune tools for ToG, can be llm (same as LLM_type), bm25 or sentencebert.
```

All the pruning and reasoning prompts utilized in the experiment are in the `prompt_list.py` file.

For eval, please see `eval/README.md` file.

## Scaling up with Yago

Note that there is a file called [main_yago_multi_thread_batched.py](./main_yago_multi_thread_batched.py) that can be used to speed up performance in case you have a large number of questions to answer. This file is a multi-threaded version of the main_yago.py file that processes the rows in batches. The number of threads and the batch size can be adjusted in the arguments provided to the file.