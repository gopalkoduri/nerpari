Relation-extraction
=================

1. Introduction
----------------

There are the following paradigms that were in vogue for extracting assertions/relations from natural language text: 
1. Traditional IE starts with a predefined vocabulary and hueristics for a domain that will help in identifying the entities and the relations between them. 
2. Open IE is a task which intends to scale relation extraction by learning about the entities and their relations, without the requirement of pre-defined vocabularies.
3. Preemptive IE
4. Ondemand IE
5. Weak supervision for IE (ontology-based learning)
6. Semantic role labeling

Unless explicitly provided with some information about a domain, most systems do not take into account the specificities of a domain when extracting relations.
We propose a framework in which linked open data can be used to supervise such relation extraction systems to improve their recall.


2. Relation extraction systems review
-------------------------------------

2.1 Open IE
-----------

1. Dependency-parse features based systems have better recall, but are slower in processing
2. POS-tag based features are faster, but might miss on non-continguous relational phrases


2.2 Traditional IE
------------------

2.3 Preemptive IE
------------------

2.4 Ondemand IE
------------------

2.5 Weak supervision for IE
---------------------------

2.6 Semantic Role Labeling
---------------------------




3. Relation types and their relevance
--------------------------------------




4. Framework
-------------


4.1 Relation extraction pipeline
-------------------------------

1. Basic cleansing of the data to keep only legible sentences.
2. Stanford corenlp tools for coref resolution
3. Open IE 4.0 and reverb systems to extract triples (arg1, relation, arg2)
4. DBpedia spotlight to map the arguments to DBpedia resources.



4.2 Relation type ranking pipeline
----------------------------------

1. Get a directed and weighted hyperlink (post using DBpedia spotlight) graph of pages from the domain in Wikipedia
2. Extract those edges which constitute the backbone of the graph or use a relevant edge centrality measure
3. Get rid of the stag nodes. For the nodes that are still left, get the DBpedia topics.
4. Create a topic\_x-topic\_y pair map and rank each pair based on the score of edges connecting the nodes in x and y.


Now, combining both the pipelines would yield us a sorted list of relation types.



References
----------

* Moro, A., & Navigli, R. (2012). WiSeNet : Building a Wikipedia-based Semantic Network with Ontologized Relations. In Conference on information and knowledge management (pp. 1672–1676).
* Fader, A., Soderland, S., & Etzioni, O. (2011). Identifying Relations for Open Information Extraction. In Empirical Methods in Natural Language Processing.
