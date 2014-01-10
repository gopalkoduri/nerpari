# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

#Run this file from the place where wiki dump is extracted

data_dir = "/homedtic/gkoduri/data/wiki/extracted"
code_dir = "/homedtic/gkoduri/workspace/ontology-learning"

# <codecell>

from glob import glob
import sys
import codecs
import pickle
from uuid import uuid5, NAMESPACE_URL
from os.path import basename
from BeautifulSoup import BeautifulSoup

# <codecell>

def build_index(arg):
    index = {}
    
    files = glob(data_dir + "/" + arg + "/wiki*")
    identifier = str(uuid5(NAMESPACE_URL, arg))
    
    log_file = code_dir + "/data/" + identifier + "_log.txt"
    log = codecs.open(log_file, "w")
    
    for f in files:
        print f
        data = codecs.open(f, 'r', 'utf-8').readlines()
        size = len(data)
        step = 100
        for ind in xrange(0, size, step):
            try:
                soup = BeautifulSoup("".join(data[ind:ind+step]))
            except UnicodeEncodeError, UnicodeDecodeError:
                log.write(f + "\t" + str(ind) + "\n")
            pages = soup.findAll('doc')
            for page in pages:
                page_title = page.attrs[2][1]
                index[page_title] = arg + "/" + basename(f)
    
    log.close()
    
    index_file = code_dir + "/data/" + identifier + ".pickle"
    pickle.dump(index, file(index_file, "w"))

# <codecell>

def merge_indexes(files):
    whole_index = {}
    for f in files:
        print f
        data = pickle.load(file(f))
        whole_index.update(data)
    
    whole_index_lower = {}
    for k, v in whole_index.items():
        whole_index_lower[k.lower()] = v
        
    return whole_index_lower

# <codecell>

if __name__ == "__main__":
    #build_index(sys.argv[1])
    
    files = glob(code_dir + "/data/wiki_index/*.pickle")
    whole_index = merge_indexes(files)
    
    pickle.dump(whole_index, file(code_dir + "/data/wiki_index.pickle", "w"))

