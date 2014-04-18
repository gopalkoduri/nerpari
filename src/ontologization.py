from __future__ import division
from collections import Counter
from numpy import concatenate

from nltk.corpus import stopwords
swords = stopwords.words('english')

from nltk.stem import PorterStemmer
stemmer = PorterStemmer()


def remove_stopwords(input_string):
    return ' '.join([i for i in input_string.strip().split() if i not in swords])


def normalize(phrase):
    return stemmer.stem(remove_stopwords(phrase.strip()))


def overlap(y, x):
    """
    %overlap of y with x
    """
    if len(x) != 0:
        return len(set(x).intersection(y))/len(x)
    else:
        return 0


def subject_relations(relations, min_num_relations=1):
    arg_subject = [normalize(i['arg1_norm']) for i in relations]
    as_c = Counter(arg_subject)
    arg_subject = [k for k, v in as_c.items() if v > min_num_relations]
    arg_subject_relations = {k: [] for k in arg_subject}

    for i in relations:
        try:
            arg_subject_relations[normalize(i['arg1_norm'])].append(normalize(i['rel_norm']))
        except KeyError:
            #these are arguments which have just one relation in the entire corpus, let them go
            continue
    return arg_subject_relations


def object_relations(relations, min_num_relations=1):
    arg_object = [normalize(i['arg2_norm']) for i in relations]
    ao_c = Counter(arg_object)
    arg_object = [k for k, v in ao_c.items() if v > min_num_relations]
    arg_object_relations = {k: [] for k in arg_object}

    for i in relations:
        try:
            arg_object_relations[normalize(i['arg2_norm'])].append(normalize(i['rel_norm']))
        except KeyError:
            #these are arguments which have just one relation in the entire corpus, let them go
            continue
    return arg_object_relations


def bootstrap(relations, seedset, num_best_relations=5):
    #get alll relations and subjects,objects
    arg_subject_relations = subject_relations(relations)
    arg_object_relations = object_relations(relations)
    
    #get the relations having seed set involved
    seed_subject_relations = concatenate([v for k, v in arg_subject_relations.items() if k in seedset])
    seed_object_relations = concatenate([v for k, v in arg_object_relations.items() if k in seedset])

    ssc = Counter(seed_subject_relations)
    soc = Counter(seed_object_relations)
    ssc.pop('')
    soc.pop('')
    print ssc
    print soc

    #Overlap scores based on relations where they are subjects
    arg_subject_scores = []

    ssc_items = sorted(ssc.items(), key=lambda x: x[1], reverse=True)
    ssc_items = [i[0] for i in ssc_items[:num_best_relations]]

    for k, v in arg_subject_relations.items():
        asc = Counter(v)
        asc_items = sorted(asc.items(), key=lambda x:x[1], reverse=True)
        asc_items = [i[0] for i in asc_items[:num_best_relations]]

        arg_subject_scores.append((k, overlap(asc_items, ssc_items)))

    arg_subject_scores = sorted(arg_subject_scores, key=lambda x: x[1], reverse=True)

    #Overlap scores based on relations where they are objects
    arg_object_scores = []

    soc_items = sorted(soc.items(), key=lambda x: x[1], reverse=True)
    soc_items = [i[0] for i in soc_items[:num_best_relations]]

    for k, v in arg_object_relations.items():
        asc = Counter(v)
        asc_items = sorted(asc.items(), key=lambda x: x[1], reverse=True)
        asc_items = [i[0] for i in asc_items[:num_best_relations]]

        arg_object_scores.append((k, overlap(asc_items, soc_items)))

    arg_object_scores = sorted(arg_object_scores, key=lambda x: x[1], reverse=True)

    return {'arg_subject_scores': arg_subject_scores, 'arg_object_scores': arg_object_scores}


































