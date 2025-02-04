from SPARQLWrapper import SPARQLWrapper, JSON
from yago_utils.constants import PREFIXES, INVALID_PROPERTIES, DB_NAME
import sys

from yago_utils.yagodb import YagoDB

SPARQLPATH = "http://localhost:8080/bigdata/sparql"  # Default path. Depends on your own internal address and port, shown in Freebase folder's readme.md
yago_db = YagoDB(DB_NAME)


def get_prefix_string() -> str:
    """
    Returns the prefixes as a substring of the SPARQL format.
    """
    prefix_list = [f"PREFIX {key}: <{value}>" for key, value in PREFIXES.items()]
    prefix_string = "\n".join(prefix_list)
    return prefix_string

PREFIX_STRING = get_prefix_string()

WD_PREFIX = "wd:"
WD_PREFIX_URL = "http://www.wikidata.org/entity/"

# This can be done because the prefixes are unique.
PREFIX_VALUES = {value: key for key, value in PREFIXES.items()}

# pre-defined sparqls
get_sparql_entities_from_labels = """
%s
SELECT DISTINCT ?entity ?label WHERE {
    VALUES ?label { %s }
    {
        ?entity rdfs:label ?label
    } UNION {
        ?entity schema:alternateName ?label
    }
    FILTER (lang(?label) = 'en')
}
"""

get_sparql_entities_from_qids = """
%s
SELECT DISTINCT ?entity ?qid WHERE {
    VALUES ?qid { %s }
    {
        ?entity owl:sameAs ?qid
    }
}
"""

def execurte_sparql(sparql_query, sparql_path = SPARQLPATH):
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

def get_entities_from_labels(labels) -> dict:
    """
    Get entities from labels.
    """
    # Get the entities from the labels
    # yago_labels: escape the labels with single quotes and @en, and deal with single quotes in the labels
    yago_labels = [f"'''{label.replace("'", "\\'")}'''@en" for label in labels]
    sparql_query = get_sparql_entities_from_labels % (PREFIX_STRING, " ".join(yago_labels))
    results = execurte_sparql(sparql_query)

    # Get the entity counts from the Yago Sqlite database
    entity_url_set = {results["entity"]["value"] for results in results}
    entities_counts = yago_db.get_entity_counts_from_labels(list(entity_url_set))
    entities = {}
    for result in results:
        entity = result["entity"]["value"]
        label = result["label"]["value"]
        if entities.get(label) is None:
            entities[label] = entity
        else:
            if (entity in entities_counts) and \
                ((entities[label] not in entities_counts) or \
                 (entities_counts[entity] > entities_counts[entities[label]])):
                entities[label] = entity

    # entities = {results["label"]["value"]: results["entity"]["value"] for results in results}
    # Replace the entities with the prefix
    replaced_entities = replace_entities_prefix(entities)
    return replaced_entities

def get_entities_from_qids(qids) -> dict:
    """
    Get entities from qids.
    """
    # Get the entities from the labels
    # yago_labels: escape the labels with single quotes and @en, and deal with single quotes in the labels
    yago_qids = [f"{WD_PREFIX}{qid}" for qid in qids]
    sparql_query = get_sparql_entities_from_qids % (PREFIX_STRING, " ".join(yago_qids))
    results = execurte_sparql(sparql_query)

    # Get wd: to yago:
    entities = {result["qid"]["value"]: result["entity"]["value"] for result in results}
    # Replace the entities with the prefix
    replaced_entities = replace_entities_prefix(entities)

    # Remove the wd: prefix from qids
    replaced_entities = {qid.replace(WD_PREFIX_URL, ""): entity for qid, entity in replaced_entities.items()}

    return replaced_entities