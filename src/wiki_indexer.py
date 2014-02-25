# -*- coding: utf-8 -*-

#Run this file from the place where wiki dump is extracted

data_dir = "/homedtic/gkoduri/data/wiki/extracted"
code_dir = "/homedtic/gkoduri/workspace/relation-extraction"

from glob import glob
import sys
import codecs
import pickle
from os.path import basename, exists
from os import mkdir
from BeautifulSoup import BeautifulSoup
import nltk
import numpy as np
import collections
from multiprocessing import Process, Manager
from gensim import corpora

from os import chdir
chdir(code_dir+"/src")
import wiki_indexer as wi
reload(wi)


#DISCONTINUED CLASS
class WikiCorpus():
    def __init__(self, data, dictionary):
        self.dictionary = dictionary
        self.data = data

    def __iter__(self):
        for tokens in self.data:
            yield self.dictionary.doc2bow(tokens)


class WikiTokenizer():
    def __init__(self, pages, wiki_index):
        self.pages = pages
        self.wiki_index = wiki_index
        self.stemming = True
        self.stopword_removal = True

        self.alphabetic_tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
        self.stemmer = nltk.stem.snowball.SnowballStemmer("english")

        self.current_page = ""
        self.content = ""
        self.tokens = []

    def __iter__(self):
        for page in self.pages:
            self.current_page = page
            self.tokenize()
            yield self.current_page, self.tokens

    def tokenize(self, page=None):
        #Tokenize the text to words
        if page:
            self.current_page = page
        if self.content == "":
            self.content = wi.get_page_content(self.current_page, self.wiki_index)

        tokenized_text = [self.alphabetic_tokenizer.tokenize(s) for s in nltk.sent_tokenize(self.content)]
        tokenized_text = np.concatenate(tokenized_text)

        #Do stemming and remove stopwords
        if self.stemming:
            tokenized_text = [self.stemmer.stem(w) for w in tokenized_text if
                              not w in nltk.corpus.stopwords.words('english')]
        elif self.stopword_removal:
            tokenized_text = [w for w in tokenized_text if not w in nltk.corpus.stopwords.words('english')]

        rare_tokens = set(w for w in set(tokenized_text) if tokenized_text.count(w) <= 1)
        short_tokens = set(w for w in set(tokenized_text) if len(w) <= 1)
        #tokenized_text = [w for w in tokenized_text if w not in list(short_tokens)]
        tokenized_text = [w for w in tokenized_text if w not in list(rare_tokens) + list(short_tokens)]

        self.tokens = tokenized_text


class WikiTokens():
    def __init__(self, tokens):
        self.tokens = tokens

    def load_tokens(self, tokens_file):
        self.tokens = pickle.load(file(tokens_file))

    def get_tokens(self, page):
        return self.tokens[page]

    def __iter__(self):
        for page, tokens in self.tokens.items():
            return page, tokens


def run(all_args, target_func, func_args=(), process_limit=8):
    count = 0
    mul_factor = 100.0/len(all_args)
    for i in xrange(0, len(all_args), process_limit):
        cur_args = all_args[i:i + process_limit]
        processes = []
        for arg in cur_args:
            cur_func_args = (arg, ) + func_args
            p = Process(target=target_func, args=cur_func_args)
            processes.append(p)
            p.start()

        count += process_limit
        sys.stdout.write("Progress: {0}%\r".format(count*mul_factor))
        sys.stdout.flush()

        for p in processes:
            p.join()


def tokenizer_instance(page, wiki_index, token_index, lock):
    tokenizer = WikiTokenizer([page], wiki_index)
    tokenizer.tokenize(page)
    with lock:
        token_index[page] = tokenizer.tokens

    return


def build_token_index(page_titles, wiki_index):
    manager = Manager()
    token_index = manager.dict()
    lock = manager.Lock()

    page_titles = [i.lower() for i in page_titles]

    run(page_titles, tokenizer_instance, (wiki_index, token_index, lock), process_limit=process_limit)

    return token_index


def link_indexer_instance(page, wiki_index, link_index, lock):
    page = page.lower()
    soup = get_page_content(page, wiki_index, textonly=False)
    if not soup:
        return
    links = soup.findAll("a")
    link_terms = [link.text.lower() for link in links]

    with lock:
        link_index[page] = np.unique(link_terms).tolist()

    return


def build_link_index(page_titles, wiki_index, process_limit=16):
    manager = Manager()
    link_index = manager.dict()
    lock = manager.Lock()

    page_titles = [i.lower() for i in page_titles]

    run(page_titles, link_indexer_instance, (wiki_index, link_index, lock), process_limit=process_limit)

    return link_index


def all_indexer_instance(page, wiki_index, link_index, token_index, lock):
    page = page.lower()
    soup = get_page_content(page, wiki_index, textonly=False)
    if not soup:
        return
    links = soup.findAll("a")
    link_terms = [link.text.lower() for link in links]

    content = " ".join(soup.findAll(text=True))
    tokenizer = WikiTokenizer([page], wiki_index)
    tokenizer.content = content
    tokenizer.tokenize(page)

    with lock:
        link_index[page] = np.unique(link_terms).tolist()
        token_index[page] = tokenizer.tokens
    return


def build_all_indexes(page_titles, wiki_index, process_limit=4):
    """
    Eg result: {"carnatic music": ["raaga", "taala", ...],
    "t. m. krishna": ["carnatic music", "vocalist" ...] ... }
    This index is directly written to a file, and not returned.
    """
    manager = Manager()
    link_index = manager.dict()
    token_index = manager.dict()
    lock = manager.Lock()

    page_titles = [i.lower() for i in page_titles]

    run(page_titles, all_indexer_instance, (wiki_index, link_index, token_index, lock), process_limit=process_limit)

    return {'link_index': link_index, 'token_index': token_index}


def build_lsa_index(token_index, f_name=None):
    dictionary = corpora.Dictionary(tokens for page, tokens in token_index.items())
    corpus = [dictionary.doc2bow(tokens) for page, tokens in token_index.items()]

    if f_name is None:
        return dictionary, corpus
    else:
        dictionary.save('/homedtic/gkoduri/workspace/relation-extraction/data/'+f_name+'.dict')
        corpora.MmCorpus.serialize('/homedtic/gkoduri/workspace/relation-extraction/data/'+f_name+'.mm', corpus)


def get_pages(music_style):
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


def get_page_content(page_title, wiki_index, textonly=True):
    page_title = page_title.lower()
    if page_title not in wiki_index.keys():
        return ""

    file_path = data_dir + "/" + wiki_index[page_title]

    data = codecs.open(file_path, 'r', 'utf-8').read()
    try:
        soup = BeautifulSoup(data)
    except (UnicodeEncodeError, UnicodeDecodeError):
        return ""

    pages = soup.findAll('doc')
    for page in pages:
        if page.attrs[2][1].lower() == page_title:
            if textonly:
                return " ".join(page.findAll(text=True))
            else:
                return page


def group_pages_by_file(pages, wiki_index):
    file_page_index = {}

    for page in pages:
        if page in wiki_index.keys():
            if wiki_index[page] in file_page_index.keys():
                file_page_index[wiki_index[page]].append(page)
            else:
                file_page_index[wiki_index[page]] = [page]

    return file_page_index


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



if __name__ == "__main__":
    all_args = sys.argv[1:]

    # Build page index
    # run(all_args, build_page_index, (), process_limit=8)

    # Build Link Indexes
    # run(all_args, build_link_index, (["hindustani", "music"], ), process_limit=8)

    # Merge Indexes
    folder = sys.argv[1].strip("/")
    files = glob(code_dir + "/data/" + folder + "/*.pickle")
    whole_index = merge_indexes(files)
    pickle.dump(whole_index, file(code_dir + "/data/" + folder + ".pickle", "w"))

    # Build Content Indexes
    # run(all_args, build_content_index, (["jazz", "music"], 30, "bigrams", True), process_limit=8)

