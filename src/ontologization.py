from __future__ import division
import numpy as np
from collections import Counter
from numpy import concatenate, array, unique, sqrt
from gensim import corpora, models, similarities
import codecs
import json

from nltk.corpus import stopwords
swords = stopwords.words('english')

from nltk.stem import PorterStemmer
stemmer = PorterStemmer()

import semantic_parsing as sp

from os.path import expanduser
home = expanduser("~")


def ochiai_coefficient(x, y):
    x = set(x)
    y = set(y)
    return len(x.intersection(y)) / (sqrt(len(x) * len(y)))


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


def get_objects(relations, min_num_relations=1, split=True, normalization=False):
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
        i[2] = i[2].lower()
        if split:
            if normalization:
                parts = normalize(i[2]).split()
            else:
                parts = i[2].split()

            for part in parts:
                try:
                    if normalization:
                        objects[i[0]].append(part)
                    else:
                        objects[i[0]].append(part)
                except KeyError:
                    #these are arguments which have just one relation in the entire corpus, let them go
                    continue
        else:
            try:
                if normalization:
                    objects[i[0]].append(normalize(i[2]))
                else:
                    objects[i[0]].append(i[2])
            except KeyError:
                #these are arguments which have just one relation in the entire corpus, let them go
                continue
    return objects


def get_transformation(data):
    temp = data.items()
    texts = [i[1] for i in temp]
    entities = [i[0] for i in temp]

    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    tfidf = models.TfidfModel(corpus)

    lsi = models.LsiModel(corpus, id2word=dictionary, num_topics=100)
    sim_index = similarities.MatrixSimilarity(lsi[tfidf[corpus]])

    return entities, dictionary, tfidf, lsi, sim_index


def iterate_lsa(seedset, objects, predicates, object_entities, object_dictionary, object_tfidf, object_lsi,
                object_sim_index, predicate_dictionary, predicate_tfidf,
                predicate_lsi, predicate_sim_index, expansion=1):
    seedset_predicates = concatenate([v for k, v in predicates.items() if k in seedset])
    seedset_objects = concatenate([v for k, v in objects.items() if k in seedset])

    sim_scores_predicates = array(predicate_sim_index[predicate_lsi[predicate_tfidf[predicate_dictionary.doc2bow(seedset_predicates)]]])
    sim_scores_objects = array(object_sim_index[object_lsi[object_tfidf[object_dictionary.doc2bow(seedset_objects)]]])
    sim_scores = dict(enumerate(0*sim_scores_predicates + 1*sim_scores_objects))
    #get rid of the scores to elements in seedset
    seedset_indices = [object_entities.index(s) for s in seedset]
    for ind in seedset_indices:
        sim_scores.pop(ind)

    sim_scores = sorted(sim_scores.items(), key=lambda x:x[1], reverse=True)
    #return sim_scores

    chosen_entities = [object_entities[i[0]] for i in sim_scores[:expansion]]
    print chosen_entities
    #print sim_scores[:expansion]
    #print
    return seedset + chosen_entities


def bootstrap_lsa(seedset, objects, predicates, expansion=1, iterations=100):
    object_entities, object_dictionary, object_tfidf, object_lsi, object_sim_index = get_transformation(objects)
    predicate_entities, predicate_dictionary, predicate_tfidf, predicate_lsi, predicate_sim_index = get_transformation(predicates)
    for i in xrange(iterations):
        seedset = iterate_lsa(seedset, objects, predicates, object_entities, object_dictionary, object_tfidf, object_lsi,
                              object_sim_index, predicate_dictionary, predicate_tfidf,
                              predicate_lsi, predicate_sim_index, expansion)
    return seedset


def iterate(seedset, entities, objects, predicates, expansion=1):
    seedset_predicates = concatenate([v for k, v in predicates.items() if k in seedset])
    seedset_objects = concatenate([v for k, v in objects.items() if k in seedset])

    #seedset predicate counter
    sp_counter = Counter(seedset_predicates)
    seedset_predicates = [p for p, count in sp_counter.items() if count > 1]

    #seedset object counter
    so_counter = Counter(seedset_objects)

    seedset_objects = [o for o, count in so_counter.items() if count > 1]
    print seedset_objects
    distances = {}
    for entity in entities:
        if entity in seedset:
            continue
        try:
            entity_predicates = list(unique(predicates[entity]))
            entity_objects = list(unique(objects[entity]))
        except KeyError:
            continue

        distance_predicates = ochiai_coefficient(seedset_predicates, entity_predicates)
        distance_objects = ochiai_coefficient(seedset_objects, entity_objects)

        distances[entity] = 0.5*distance_predicates + 0.5*distance_objects

    distances = sorted(distances.items(), key=lambda x: x[1], reverse=True)

    chosen_entities = [i[0] for i in distances[:expansion]]
    print distances[:expansion]
    print
    return seedset + chosen_entities


def bootstrap(seedset, entities, objects, predicates, expansion=1, iterations=100):
    for i in xrange(iterations):
        seedset = iterate(seedset, entities, objects, predicates, expansion)
    return seedset


def get_groundtruth(music, category):
    groundtruth_file = home + '/workspace/nerpari/data/wiki_pages/' + music + '_' + category + '.json'
    data = json.load(file(groundtruth_file))
    data = [i['s']['value'] for i in data['results']['bindings']]
    data = [i.split('/')[-1].replace('_', ' ').lower() for i in data]
    return set(data)


if __name__ == "__main__":
    artist_seedset = ['Gayathri Venkataraghavan', 'Sanjay Subrahmanyan', 'Abhishek Raghuram']
    composer_seedset = ['Syama Sastri', 'Muthuswami Dikshitar', 'Tyagaraja']
    raaga_seedset = ['Charukesi', 'Shuddha Saveri', 'Abhogi']
    instrument_seedset = ['Mridangam', 'Venu']
    form_seedset = ['Viruttam', 'Varnam', 'Niraval']

    data = codecs.open(home+'/workspace/nerpari/data/ambati/data/carnatic_music/resolved-sentences-unidecoded-sem.txt', encoding='utf-8').readlines()

    relations = []
    #progress = progressbar.ProgressBar(len(data))

    for ind in np.arange(0, len(data), 20):
        rg = sp.get_graph(data, ind, draw=False)
        res = sp.get_relations(rg)
        res = sp.expand_relations(rg, res)
        for rel in res['valid']:
            relations.append(rel)
        #progress.animate(ind)

    wiki_entities = codecs.open(home + '/workspace/nerpari/data/wiki_pages/carnatic_music_pages.txt', encoding='utf-8').readlines()
    wiki_entities = [i.strip() for i in wiki_entities]

    filtered_relations = []
    for rel in relations:
        if rel[0] in wiki_entities:# or rel[2] in wiki_entities:
            filtered_relations.append(rel)

    predicates = get_predicates(filtered_relations, normalization=False)
    objects = get_objects(filtered_relations, split=True, normalization=True)

    res = bootstrap_lsa(artist_seedset, objects, predicates, expansion=1, iterations=10)