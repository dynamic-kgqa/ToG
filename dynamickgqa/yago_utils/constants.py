# For Yago Sqlite3 database
DB_NAME = "yago_utils/yago_all.db"


# For Yago SPARQL KG
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

INVALID_PROPERTIES = {
    # "schema:image",
    # "schema:mainEntityOfPage",
    # "schema:dateCreated",
    # "schema:iataCode",
    # "yago:iswcCode",
    "owl:sameAs",
    # "yago:unemploymentRate",
    # "schema:isbn",
    # "schema:postalCode",
    # "schema:icaoCode",
    # "schema:url",
    # "rdfs:comment",
    # "schema:leiCode",
    # "yago:humanDevelopmentIndex",
    # "yago:length",
    # "schema:gtin",
    # "schema:logo",
    # "schema:geo",
    "schema:sameAs"
}