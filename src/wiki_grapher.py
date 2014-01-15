# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

data_dir = "/homedtic/gkoduri/data/wiki/extracted"
code_dir = "/homedtic/gkoduri/workspace/relation-extraction"

# <codecell>

from BeautifulSoup import BeautifulSoup
from glob import glob
from math import sqrt, floor
import pickle
import codecs
import networkx as nx
import sys

# <codecell>

def graph_from_link_index(wiki_link_index, weight_thresh):
    pages = wiki_link_index.keys()
    n = len(pages)
    g = nx.Graph()
    
    total_calc = floor(n*(n-1)/2)
    mul_factor = 100.0/total_calc
    count = 0
    for i in xrange(0, n):
        for j in xrange(i+1, n):
            x = set(wiki_link_index[pages[i]])
            y = set(wiki_link_index[pages[j]])
            
            #Ochiai coefficient, in this case is equal to Cosine similarity
            if len(x) == 0 or len(y) == 0:
                weight = 0
            else:
                if pages[j] in x or pages[i] in y:
                    weight = (1+len(x.intersection(y)))/(sqrt(len(x)*len(y)))
                else:
                    weight = len(x.intersection(y))/(sqrt(len(x)*len(y)))
            #print pages[i], pages[j], weight
            if weight >= weight_thresh:
                g.add_edge(pages[i], pages[j], weight=weight)
        count += (n-i-1)
        sys.stdout.write("Progress: {0}%\r".format(count*mul_factor))
        sys.stdout.flush()
    return g

# <codecell>

if __name__ == "__main__":
    pass

