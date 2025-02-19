extract_relation_prompt = """Please retrieve %s relations (separated by semicolon) that contribute to the question and rate their contribution on a scale from 0 to 1 (the sum of the scores of %s relations is 1).
Q: Name the president of the country whose main spoken language was Brahui in 1980?
Topic Entity: Brahui Language
Relations: yago:officialLanguage; schema:knowsLanguage; schema:inLanguage; schema:contentLocation; schema:location; yago:capital; yago:leader; yago:neighbors; schema:birthPlace; schema:address; schema:about
A: {yago:officialLanguage (Score: 0.4)}: This relation is highly relevant as it directly identifies whether Brahui was an official language of a country in 1980, which helps in determining the country and its president.
2. {schema:knowsLanguage (Score: 0.3)}: This relation provides insight into entities (people, organizations, regions) that are associated with Brahui, helping verify its primary country of use.
3. {schema:location (Score: 0.2)}: This relation provides geographical information about the locations where Brahui was spoken, further narrowing down the country of interest.

Q: """

score_entity_candidates_prompt = """Please score the entities' contribution to the question on a scale from 0 to 1 (the sum of the scores of all entities is 1).
Q: The movie featured Miley Cyrus and was produced by Tobin Armbrust?
Relation: schema:producer
Entites: The Resident; So Undercover; Let Me In; Begin Again; The Quiet Ones; A Walk Among the Tombstones
Score: 0.0, 1.0, 0.0, 0.0, 0.0, 0.0
The movie that matches the given criteria is "So Undercover" with Miley Cyrus and produced by Tobin Armbrust. Therefore, the score for "So Undercover" would be 1, and the scores for all other entities would be 0.

Q: {}
Relation: {}
Entites: """

answer_prompt = """Given a question and the associated retrieved knowledge graph triplets (entity, relation, entity), you are asked to answer the question with these triplets and your knowledge.
Q: Find the person who said \"Taste cannot be controlled by law\", what did this person die from?
Knowledge Triplets: Taste cannot be controlled by law., schema:author, Thomas Jefferson
A: Based on the given knowledge triplets, it's not sufficient to answer the entire question. The triplets only provide information about the person who said "Taste cannot be controlled by law," which is Thomas Jefferson. To answer the second part of the question, it's necessary to have additional knowledge about where Thomas Jefferson's dead.

Q: The artist nominated for The Long Winter was born where?
Knowledge Triplets: The Long Winter, schema:author, Laura Ingalls Wilder
Laura Ingalls Wilder, schema:birthPlace, Pepin County
Pepin County, schema:location, Wisconsin
A: Based on the given knowledge triplets, the author of The Long Winter, Laura Ingalls Wilder, lived in Pepin County, Wisconsin. Therefore, the answer to the question is {Pepin County}.

Q: Who is the coach of the team owned by Steve Bisciotti?
Knowledge Triplets: Steve Bisciotti, sports.professional_sports_team.owner_s, Baltimore Ravens
Steve Bisciotti, schema:owns, Baltimore Ravens
Steve Bisciotti, schema:founder, Allegis Group
A: Based on the given knowledge triplets, the coach of the team owned by Steve Bisciotti is not explicitly mentioned. However, it can be inferred that the team owned by Steve Bisciotti is the Baltimore Ravens, a professional sports team. Therefore, additional knowledge about the current coach of the Baltimore Ravens can be used to answer the question.

Q: Rift Valley Province is located in a nation that uses which form of currency?
Knowledge Triplets: Rift Valley Province, schema:location, Kenya
Kenya, yago:officialLanguage, Swahili
Kenya, schema:currency, Kenyan Shilling
A: Based on the given knowledge triplets, Rift Valley Province is located in Kenya, which uses the Kenyan shilling as its currency. Therefore, the answer to the question is {Kenyan shilling}.

Q: The country with the leader as Narendra Modi borders which nations?
Knowledge Triplets: Narendra Modi, yago:leader, India
Narendra Modi, schema:birthPlace, Vadnagar
Vadnagar, schema:location, India
A: Based on the given knowledge triplets, Narendra Modi is the leader of India. However, the triplets do not explicitly mention the countries that border India. To answer the question, additional knowledge about India’s neighboring countries is required.

Q: {}
"""

prompt_evaluate="""Given a question and the associated retrieved knowledge graph triplets (entity, relation, entity), you are asked to answer whether it's sufficient for you to answer the question with these triplets and your knowledge (Yes or No).
Q: Find the person who said \"Taste cannot be controlled by law\", what did this person die from?
Knowledge Triplets: Taste cannot be controlled by law, schema:author, Thomas Jefferson
A: {No}. Based on the given knowledge triplets, it's not sufficient to answer the entire question. The triplets only provide information about the person who said "Taste cannot be controlled by law," which is Thomas Jefferson. To answer the second part of the question, it's necessary to have additional knowledge about where Thomas Jefferson's dead.

Q: The artist nominated for The Long Winter was born where?
Knowledge Triplets: The Long Winter, schema:author, Laura Ingalls Wilder
Laura Ingalls Wilder, schema:birthPlace, Pepin County
Pepin County, schema:location, Wisconsin
A: {Yes}. Based on the given knowledge triplets, the author of The Long Winter, Laura Ingalls Wilder, lived in Pepin County, Wisconsin. Therefore, the answer to the question is {Pepin County}.

Q: Who is the coach of the team owned by Steve Bisciotti?
Knowledge Triplets: Steve Bisciotti, sports.professional_sports_team.owner_s, Baltimore Ravens
Steve Bisciotti, schema:owns, Baltimore Ravens
Steve Bisciotti, schema:founder, Allegis Group
A: {No}. Based on the given knowledge triplets, the coach of the team owned by Steve Bisciotti is not explicitly mentioned. However, it can be inferred that the team owned by Steve Bisciotti is the Baltimore Ravens, a professional sports team. Therefore, additional knowledge about the current coach of the Baltimore Ravens can be used to answer the question.

Q: Rift Valley Province is located in a nation that uses which form of currency?
Knowledge Triplets: Rift Valley Province, schema:location, Kenya
Kenya, yago:officialLanguage, Swahili
Kenya, schema:currency, Kenyan Shilling
A: {Yes}. Based on the given knowledge triplets, Rift Valley Province is located in Kenya, which uses the Kenyan shilling as its currency. Therefore, the answer to the question is {Kenyan shilling}.

Q: The country with the leader as Narendra Modi borders which nations?
Knowledge Triplets: Narendra Modi, yago:leader, India
Narendra Modi, schema:birthPlace, Vadnagar
Vadnagar, schema:location, India
A: {No}. Based on the given knowledge triplets, Narendra Modi is the leader of India. However, the triplets do not explicitly mention the countries that border India. To answer the question, additional knowledge about India’s neighboring countries is required.

"""

cot_prompt = """Q: What state is home to the university that is represented in sports by George Washington Colonials men's basketball?
A: First, the education institution has a sports team named George Washington Colonials men's basketball in is George Washington University , Second, George Washington University is in Washington D.C. The answer is {Washington, D.C.}.

Q: Who lists Pramatha Chaudhuri as an influence and wrote Jana Gana Mana?
A: First, Bharoto Bhagyo Bidhata wrote Jana Gana Mana. Second, Bharoto Bhagyo Bidhata lists Pramatha Chaudhuri as an influence. The answer is {Bharoto Bhagyo Bidhata}.

Q: Who was the artist nominated for an award for You Drive Me Crazy?
A: First, the artist nominated for an award for You Drive Me Crazy is Britney Spears. The answer is {Jason Allen Alexander}.

Q: What person born in Siegen influenced the work of Vincent Van Gogh?
A: First, Peter Paul Rubens, Claude Monet and etc. influenced the work of Vincent Van Gogh. Second, Peter Paul Rubens born in Siegen. The answer is {Peter Paul Rubens}.

Q: What is the country close to Russia where Mikheil Saakashvii holds a government position?
A: First, China, Norway, Finland, Estonia and Georgia is close to Russia. Second, Mikheil Saakashvii holds a government position at Georgia. The answer is {Georgia}.

Q: What drug did the actor who portrayed the character Urethane Wheels Guy overdosed on?
A: First, Mitchell Lee Hedberg portrayed character Urethane Wheels Guy. Second, Mitchell Lee Hedberg overdose Heroin. The answer is {Heroin}."""
