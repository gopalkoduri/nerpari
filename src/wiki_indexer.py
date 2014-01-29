# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

#Run this file from the place where wiki dump is extracted

data_dir = "/homedtic/gkoduri/data/wiki/extracted"
code_dir = "/homedtic/gkoduri/workspace/relation-extraction"

# <codecell>

from glob import glob
import sys
import codecs
import pickle
from uuid import uuid5, NAMESPACE_URL
from os.path import basename, exists
from os import mkdir
from BeautifulSoup import BeautifulSoup
import nltk
import numpy as np
import collections
from multiprocessing import Process

# <codecell>g


def build_page_index(wiki_folder_arg):
    """
    After we run wiki_extractor script on the wiki dump, we get a folder structure.
    Given one folder (eg: AA), this function builds an index of pages. The index's
    keys are page titles, and values are the file path from the given folder, which
    contain that page.

    Eg: {"carnatic music": "AA/wiki_35", "t. m. krishna": "BD/wiki_89" ... }
    This index is directly written to a file, and not returned.
    """

    index = {}
    
    files = glob(data_dir + "/" + wiki_folder_arg + "/wiki*")

    if not exists(code_dir + "/data/wiki_index/"):
        mkdir(code_dir + "/data/wiki_index/")

    log_file = code_dir + "/data/wiki_index/" + wiki_folder_arg + "_log.txt"
    log = codecs.open(log_file, "w")
    
    for f in files:
        print f
        data = codecs.open(f, 'r', 'utf-8').readlines()
        size = len(data)
        step = 100
        for ind in xrange(0, size, step):
            try:
                soup = BeautifulSoup("".join(data[ind:ind+step]))
            except (UnicodeDecodeError, UnicodeEncodeError):
                log.write(f + "\t" + str(ind) + "\n")
                continue
            pages = soup.findAll('doc')
            for page in pages:
                page_title = page.attrs[2][1]
                index[page_title.lower()] = wiki_folder_arg + "/" + basename(f)
    
    log.close()
    
    index_file = code_dir + "/data/wiki_index/" + wiki_folder_arg + ".pickle"
    pickle.dump(index, file(index_file, "w"))

# <codecell>


def build_link_index(wiki_folder_arg, music_style):
    """
    After we run wiki_extractor script on the wiki dump, we get a folder structure.
    Given one folder (eg: AA), and a keyword that describes a given music style, this
    function filters those pages which have the keyword music_style, and builds an
    index of all such pages. The keys are page titles and values are vectors of page titles
    which are hyperlinked from the given page.

    Eg: {"carnatic music": ["raaga", "taala", ...],
    "t. m. krishna": ["carnatic music", "vocalist" ...] ... }
    This index is directly written to a file, and not returned.
    """
    link_index = {}
    
    files = glob(data_dir + "/" + wiki_folder_arg + "/wiki*")

    if not exists(code_dir + "/data/" + music_style + "_links"):
        mkdir(code_dir + "/data/" + music_style + "_links")

    log_file = code_dir + "/data/" + music_style + "_links" + wiki_folder_arg + "_log.txt"
    log = codecs.open(log_file, "w")
    
    for f in files:
        print f
        data = codecs.open(f, 'r', 'utf-8').read()
        try:
            soup = BeautifulSoup(data.lower())
        except (UnicodeEncodeError, UnicodeDecodeError):
            log.write(f + "\n")
            continue
        pages = soup.findAll('doc')
        for page in pages:
            if music_style not in page.text:
                continue
            page_title = page.attrs[2][1]
            links = page.findAll("a")
            link_terms = [link.text.lower() for link in links]
            link_index[page_title.lower()] = np.unique(link_terms).tolist()
    
    log.close()
    
    link_index_file = code_dir + "/data/" + music_style + "_links" + wiki_folder_arg + ".pickle"
    pickle.dump(link_index, file(link_index_file, "w+"))


def build_content_index(wiki_folder_arg, music_style, num_features=30, method="bigrams", stemming=True):
    """
    After we run wiki_extractor script on the wiki dump, we get a folder structure.
    Given one folder (eg: AA), and a keyword that describes a given music style, this
    function filters those pages which have the keyword music_style, and builds an
    index of all such pages. The keys are page titles and values are the most relevant
    n-grams of words.

    Eg: {"carnatic music": ["raaga", "taala", ...],
    "t. m. krishna": ["carnatic music", "vocalist" ...] ... }
    This index is directly written to a file, and not returned.

    The method argument can be either bigrams or unigrams.
    """
    #Initialize everything
    alphabetic_tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
    stemmer = nltk.stem.LancasterStemmer()

    content_index = {}

    files = glob(data_dir + "/" + wiki_folder_arg + "/wiki*")

    #Logging
    if not exists(code_dir + "/data/" + music_style + "_" + method):
        mkdir(code_dir + "/data/" + music_style + "_" + method)
    log_file = code_dir + "/data/" + music_style + "_" + method + "/" + wiki_folder_arg + ".txt"
    log = codecs.open(log_file, "w")

    for f in files:
        print f

        #Try reading data from the wiki file
        data = codecs.open(f, 'r', 'utf-8').read()
        try:
            soup = BeautifulSoup(data.lower())
        except (UnicodeEncodeError, UnicodeDecodeError):
            log.write(f + "\n")
            continue

        #Get all the pages
        pages = soup.findAll('doc')
        for page in pages:

            #Get plain text from the page
            plain_text = " ".join(page.findAll(text=True))
            if music_style not in plain_text:
                continue
            page_title = page.attrs[2][1]

            #Tokenize the text to words
            tokenized_text = [alphabetic_tokenizer.tokenize(s) for s in nltk.sent_tokenize(plain_text)]
            tokenized_text = np.concatenate(tokenized_text)

            #Do stemming and remove stopwords
            if stemming:
                tokenized_text = [stemmer.stem(w) for w in tokenized_text if not w in nltk.corpus.stopwords.words('english')]
            else:
                tokenized_text = [w for w in tokenized_text if not w in nltk.corpus.stopwords.words('english')]

            if method == "bigrams":
                bigram_measures = nltk.collocations.BigramAssocMeasures()
                finder = nltk.collocations.BigramCollocationFinder.from_words(tokenized_text)
                bigrams = finder.nbest(bigram_measures.raw_freq, num_features)

                content_index[page_title.lower()] = [" ".join(i) for i in bigrams]
            elif method == "unigrams":
                counter = collections.Counter(tokenized_text)
                unigrams = [i[0] for i in counter.most_common(num_features)]

                content_index[page_title.lower()] = unigrams
            else:
                print "\n\tThis method is not implemented. Try bigrams or unigrams.\n"
                exit()
    log.close()

    content_index_file = code_dir + "/data/" + music_style + "_" + method + "/" + wiki_folder_arg + ".pickle"
    pickle.dump(content_index, file(content_index_file, "w+"))


def get_pages_musicstyle(music_style):
    """
    Get those pages which have musicstyle keyword in them.
    """

    files = glob(data_dir + "/*/wiki*")
    rel_pages = []

    for f in files:
        print f
        data = codecs.open(f, 'r', 'utf-8').read()
        try:
            soup = BeautifulSoup(data.lower())
        except (UnicodeEncodeError, UnicodeDecodeError):
            print f
            continue

        pages = soup.findAll('doc')
        for page in pages:
            plain_text = " ".join(page.findAll(text=True))
            if music_style not in plain_text:
                continue
            rel_pages.append(page)
    return rel_pages


def group_pages_by_file(pages, wiki_index):
    file_page_index = {}

    for page in pages:
        if page in wiki_index.keys():
            if wiki_index[page] in file_page_index.keys():
                file_page_index[wiki_index[page]].append(page)
            else:
                file_page_index[wiki_index[page]] = [page]

    return file_page_index


# <codecell>

def merge_indexes(files):
    """
    Give a list of indexes from either of the above two functions,
    and it merges all of them.
    """
    whole_index = {}
    for f in files:
        print f
        data = pickle.load(file(f))
        whole_index.update(data)
        
    return whole_index

# <codecell>

if __name__ == "__main__":
    # Build page index
    # build_page_index(sys.argv[1])

    # Build Link Indexes
    # import sys
    # for folder in sys.argv[1:]:
    #    build_link_index(folder, "jazz")

    # Merge Indexes
    # files = glob(code_dir + "/data/wiki_content_index_carnatic/*.pickle")
    # whole_index = merge_indexes(files)
    # pickle.dump(whole_index, file(code_dir + "/data/wiki_content_index_carnatic.pickle", "w"))

    # Build Content Indexes

    process_limit = 8
    all_args = sys.argv[1:]
    for i in xrange(0, len(all_args), process_limit):
        cur_args = all_args[i:i+process_limit]
        processes = []
        for folder in cur_args:
            p = Process(target=build_content_index, args=(folder, "jazz", 30, "unigrams", True))
            processes.append(p)
            p.start()

        for p in processes:
            p.join()