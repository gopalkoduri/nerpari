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


def get_predicates(relations, min_num_relations=1, normalization=False):
    """
    This function returns a dictionary of subjects and their predicates in
    the form of {subject: predicates} given a set of relations.

    relations is a list of tuples, each tuple a triple (s, p, o).
    min_num_relations is the min num of relations a subject should be seen in
    to be included in results.
    normalization, is set to True, removes stop words and lemmatizes the predicates.
    """
    subjects = [i[0] for i in relations]
    subjects_counter = Counter(subjects)
    subjects = [k for k, v in subjects_counter.items() if v > min_num_relations]

    predicates = {k: [] for k in subjects}
    for i in relations:
        try:
            if normalization:
                predicates[i[0]].append(normalize(i[1]))
            else:
                predicates[i[0]].append(i[1])
        except KeyError:
            #these are arguments which have just one relation in the entire corpus, let them go
            continue
    return predicates


def get_objects(relations, min_num_relations=1, normalization=False):
    """
    This function returns a dictionary of subjects and their predicates in
    the form of {subject: objects} given a set of relations.

    relations is a list of tuples, each tuple a triple (s, p, o).
    min_num_relations is the min num of relations a object should be seen in
    to be included in results.
    normalization, is set to True, removes stop words and lemmatizes the objects.
    """
    subjects = [i[0] for i in relations]
    subjects_counter = Counter(subjects)
    subjects = [k for k, v in subjects_counter.items() if v > min_num_relations]

    objects = {k: [] for k in subjects}
    for i in relations:
        try:
            if normalization:
                objects[i[0]].append(normalize(i[2]))
            else:
                objects[i[0]].append(i[2])
        except KeyError:
            #these are arguments which have just one relation in the entire corpus, let them go
            continue
    return objects


def bootstrap(relations, seedset, num_best_relations=5):
    #get alll relations and subjects,objects
    arg_subject_relations = get_predicates(relations)
    arg_object_relations = get_objects(relations)
    
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


































