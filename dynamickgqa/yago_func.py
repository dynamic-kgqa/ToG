from SPARQLWrapper import SPARQLWrapper, JSON
from yago_utils.constants import PREFIXES, INVALID_PROPERTIES
import sys

SPARQLPATH = "http://localhost:8080/bigdata/sparql"  # Default path. Depends on your own internal address and port, shown in Freebase folder's readme.md

def get_prefix_string() -> str:
    """
    Returns the prefixes as a substring of the SPARQL format.
    """
    prefix_list = [f"PREFIX {key}: <{value}>" for key, value in PREFIXES.items()]
    prefix_string = "\n".join(prefix_list)
    return prefix_string

PREFIX_STRING = get_prefix_string()

# This can be done because the prefixes are unique.
PREFIX_VALUES = {value: key for key, value in PREFIXES.items()}

# pre-defined sparqls
get_sparql_entities_from_labels = """
%s
SELECT DISTINCT ?entity ?label WHERE {
    VALUES ?label { %s }
    ?entity rdfs:label ?label
}
"""

def execurte_sparql(sparql_query, sparql_path = SPARQLPATH):
    print(sparql_query)
    sparql = SPARQLWrapper(sparql_path)
    sparql.setQuery(sparql_query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    return results["results"]["bindings"]

def replace_entities_prefix(entities):
    """
    Replaces the prefix value (URL) with the prefix key.
    """
    # Instead of removing the entire prefix url, we just replace with the prefix key.
    replaced_entities = {}
    for entity_label, entity in entities.items():
        for prefix, value in PREFIXES.items():
            if entity.startswith(value):
                replaced_entities[entity_label] = entity.replace(value, f"{prefix}:")
                break
    return replaced_entities

def get_entities_from_labels(labels):
    """
    Get entities from labels.
    """
    yago_labels = [f"\'{label}\'@en" for label in labels]
    sparql_query = get_sparql_entities_from_labels % (PREFIX_STRING, " ".join(yago_labels))
    results = execurte_sparql(sparql_query)
    entities = {results["label"]["value"]: results["entity"]["value"] for results in results}
    replaced_entities = replace_entities_prefix(entities)
    return replaced_entities