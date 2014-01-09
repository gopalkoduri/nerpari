Ontology-learning
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


2. Methods
----------

2.1 Open IE
-----------

1. Dependency-parse features based systems have better recall, but are slower in processing
2. POS-tag based features are faster, but might miss on non-continguous relational phrases


2.1.1 Steps
----------

1. Get the sentences
2. Use openie-4.0 (or a similar system) to get (s, relational phrase, o) triples.
   Also try dependency parse features to compare with the results of reverb.
3. Use WiseNet to disambiguate the relational phrases


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



References
----------

* Moro, A., & Navigli, R. (2012). WiSeNet : Building a Wikipedia-based Semantic Network with Ontologized Relations. In Conference on information and knowledge management (pp. 1672–1676).
* Fader, A., Soderland, S., & Etzioni, O. (2011). Identifying Relations for Open Information Extraction. In Empirical Methods in Natural Language Processing.
