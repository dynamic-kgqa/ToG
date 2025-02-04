"""
This standalone script is used to generate answer alias file for DynamicKGQA dataset.

This file is more useful for PoG than ToG. The PoG model uses the alias of the answer to generate the answer.
"""
import json
import argparse
from datasets import load_dataset
from SPARQLWrapper import SPARQLWrapper, JSON

HF_PATH = "preetam7/dynamic_kgqa"
JSON_FILE = "dynamickgqa_test.json"
ALIAS_JSONL_OUTPUT_FILE = "dynamickgqa_test_alias.jsonl"
ALIAS_FINAL_OUTPUT_FILE = "dynamickgqa_test_alias.json"
ANSWER_URI_KEY = "answer_uri"

SPARQLPATH = "http://localhost:8080/bigdata/sparql"

PREFIXES = {
    "yago": "http://yago-knowledge.org/resource/",
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "xsd": "http://www.w3.org/2001/XMLSchema#",
    "ontolex": "http://www.w3.org/ns/lemon/ontolex#",
    "dct": "http://purl.org/dc/terms/",
    "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
    "owl": "http://www.w3.org/2002/07/owl#",
    "wikibase": "http://wikiba.se/ontology#",
    "skos": "http://www.w3.org/2004/02/skos/core#",
    "schema": "http://schema.org/",
    "cc": "http://creativecommons.org/ns#",
    "geo": "http://www.opengis.net/ont/geosparql#",
    "prov": "http://www.w3.org/ns/prov#",
    "wd": "http://www.wikidata.org/entity/",
    "data": "https://www.wikidata.org/wiki/Special:EntityData/",
    "sh": "http://www.w3.org/ns/shacl#",
    "s": "http://www.wikidata.org/entity/statement/",
    "ref": "http://www.wikidata.org/reference/",
    "v": "http://www.wikidata.org/value/",
    "wdt": "http://www.wikidata.org/prop/direct/",
    "wpq": "http://www.wikidata.org/prop/quant/",
    "wdtn": "http://www.wikidata.org/prop/direct-normalized/",
    "p": "http://www.wikidata.org/prop/",
    "ps": "http://www.wikidata.org/prop/statement/",
    "psv": "http://www.wikidata.org/prop/statement/value/",
    "psn": "http://www.wikidata.org/prop/statement/value-normalized/",
    "pq": "http://www.wikidata.org/prop/qualifier/",
    "pqv": "http://www.wikidata.org/prop/qualifier/value/",
    "pqn": "http://www.wikidata.org/prop/qualifier/value-normalized/",
    "pr": "http://www.wikidata.org/prop/reference/",
    "prv": "http://www.wikidata.org/prop/reference/value/",
    "prn": "http://www.wikidata.org/prop/reference/value-normalized/",
    "wdno": "http://www.wikidata.org/prop/novalue/",
    "ys": "http://yago-knowledge.org/schema#"
}

def execurte_sparql(sparql_query, sparql_path = SPARQLPATH):
    sparql = SPARQLWrapper(sparql_path)
    sparql.setQuery(sparql_query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    return results["results"]["bindings"]

def get_prefix_string() -> str:
    """
    Returns the prefixes as a substring of the SPARQL format.
    """
    prefix_list = [f"PREFIX {key}: <{value}>" for key, value in PREFIXES.items()]
    prefix_string = "\n".join(prefix_list)
    return prefix_string

PREFIX_STRING = get_prefix_string()

def alias_query(answers):
    """
    Uses the Sparql query to get the alias of the answer.
    """
    answers = [f"<{answer}>" for answer in answers]
    query = f"""
    {PREFIX_STRING}
    SELECT ?answer ?alias WHERE {{
        VALUES ?answer {{ {" ".join(answers)} }}
        {{
            ?answer schema:alternateName ?alias
            FILTER(lang(?alias) = "en")
        }} UNION {{
            ?answer rdfs:label ?alias
            FILTER(lang(?alias) = "en")
        }}
    }}
    """
    return query

def convert_to_json(jsonl_file, json_file):
    """
    Converts the jsonl file to json file.
    """
    with open(jsonl_file, "r") as f:
        data = f.readlines()
    
    json_data = {}
    for index, line in enumerate(data):
        line = json.loads(line)
        for key in line:
            if json_data.get(key) is None: json_data[key] = []
            json_data[key] = line[key]
        
    with open(json_file, "w") as f:
        json.dump(json_data, f)

def main():
    # Load json dataset
    with open(JSON_FILE, "r") as f:
        data = json.load(f)
    
    answers = []
    threshold = 10
    for index, item in enumerate(data):
        answers.append(item[ANSWER_URI_KEY])
        if (index + 1) % threshold == 0:
            query = alias_query(answers)
            results = execurte_sparql(query)

            aliases = {}
            for result in results:
                answer = result["answer"]["value"]
                alias = result["alias"]["value"]
                if aliases.get(answer) is None: aliases[answer] = []
                aliases[answer].append(alias)

            # Save to the output jsonl file
            with open(ALIAS_JSONL_OUTPUT_FILE, "a+") as f:
                for answer, alias in aliases.items():
                    f.write(json.dumps({"answer": answer, "alias": alias}) + "\n")

            answers = []
            print(f"Processed {index + 1} answers")

if __name__ == "__main__":
    # main()
    convert_to_json(ALIAS_JSONL_OUTPUT_FILE, ALIAS_FINAL_OUTPUT_FILE)