import re
import codecs

from os.path import expanduser
home = expanduser("~")

import sys
sys.path.append(home + '/workspace')


def get_openie_relations(input_file):
    """
    Each line in the input_file will have 6 fields when split by a tab.

    0. Confidence score
    1. Context (i.e., condition or something that is like a reification)
    2. Argument 1
    3. Relation
    4. Argument 2, 3 ... (Simple/Spatial/Temporal)
    5. The entire input sentence
    """
    relations = codecs.open(input_file, encoding='utf-8').readlines()

    arg_starts = ['SimpleArgument\(', 'SpatialArgument\(', 'TemporalArgument\(']
    rel_start = 'Relation\('
    end = ',List\('

    relations_parsed = []

    for rel_data in relations:
        rel_parts = rel_data.split('\t')

        #We are skipping those relations which have some reification kind of context
        if rel_parts[1]:
            continue

        #Confidence score
        confidence = float(rel_parts[0])

        #First argument
        expr = arg_starts[0] + '(.*)' + end
        arg1 = re.search(expr, rel_parts[2])
        if arg1:
            arg1 = arg1.group(1)
        else:
            continue

        #Relation
        expr = rel_start + '(.*)' + end
        rel_string = re.search(expr, rel_parts[3])
        if rel_string:
            rel_string = rel_string.group(1)
        else:
            continue

        #Second argument, can be multiple ...
        arg2 = []
        temp = rel_parts[4].split(');')
        for chunk in temp:
            for arg_start in arg_starts:
                expr = arg_start + '(.*)' + end
                arg = re.search(expr, chunk)
                if arg:
                    arg2.append(arg.group(1))

        # ... so, we split each argument in a relation
        for arg in arg2:
            rel_dict = {'arg1': arg1, 'rel': rel_string, 'arg2': arg,
                        'confidence': confidence, 'full_sentence': rel_parts[-1].strip()}
            relations_parsed.append(rel_dict)

    return relations_parsed


def get_reverb_relations():
    pass