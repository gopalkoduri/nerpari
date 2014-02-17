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
from multiprocessing import Process
from gensim import corpora

from os import chdir
chdir(code_dir+"/src")
import wiki_indexer as wi
reload(wi)


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
            self.content = wi.get_page_content(page, self.wiki_index)
            self.tokenize()
            yield self.current_page, self.tokens

    def tokenize(self):
        #Tokenize the text to words
        tokenized_text = [self.alphabetic_tokenizer.tokenize(s) for s in nltk.sent_tokenize(self.content)]
        tokenized_text = np.concatenate(tokenized_text)

        #Do stemming and remove stopwords
        if self.stemming:
            tokenized_text = [self.stemmer.stem(w) for w in tokenized_text if
                              not w in nltk.corpus.stopwords.words('english')]
        elif self.stopword_removal:
            tokenized_text = [w for w in tokenized_text if not w in nltk.corpus.stopwords.words('english')]

        rare_tokens = set(w for w in set(tokenized_text) if tokenized_text.count(w) == 1)
        tokenized_text = [w for w in tokenized_text if w not in rare_tokens]

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


#DISCONTINUED CLASS
class WikiCorpus():
    def __init__(self, data, dictionary):
        self.dictionary = dictionary
        self.data = data

    def __iter__(self):
        for tokens in self.data:
            yield self.dictionary.doc2bow(tokens)


def build_token_index(page_titles, wiki_index):
    token_index = {}
    data = WikiTokenizer(page_titles, wiki_index)
    count = 0
    mul_factor = 100.0/len(page_titles)
    for page, tokens in data:
        count += 1
        token_index[page] = tokens
        sys.stdout.write("Progress: {0}%\r".format(count*mul_factor))
        sys.stdout.flush()

    return token_index


def build_lsa_index(token_index, folder_arg=None):
    dictionary = corpora.Dictionary(tokens for page, tokens in token_index.items())
    corpus = [dictionary.doc2bow(tokens) for page, tokens in token_index.items()]

    if folder_arg is None:
        return dictionary, corpus
    else:
        dictionary.save('/homedtic/gkoduri/workspace/relation-extraction/data/content-analysis/'
                        +folder_arg+'/'+folder_arg+'.dict')
        corpora.MmCorpus.serialize('/homedtic/gkoduri/workspace/relation-extraction/data/content-analysis/'
                                   +folder_arg+'/'+folder_arg+'.mm', corpus)


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
                soup = BeautifulSoup("".join(data[ind:ind + step]))
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


def build_link_index(wiki_folder_arg, keywords):
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
    keywords = [i.lower() for i in keywords]

    files = glob(data_dir + "/" + wiki_folder_arg + "/wiki*")

    if not exists(code_dir + "/data/" + "_".join(keywords) + "_hyperlinks"):
        mkdir(code_dir + "/data/" + "_".join(keywords) + "_hyperlinks")

    log_file = code_dir + "/data/" + "_".join(keywords) + "_hyperlinks/" + wiki_folder_arg + "_log.txt"
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
            plain_text = " ".join(page.findAll(text=True))
            good_to_go = True
            for keyword in keywords:
                if keyword not in plain_text:
                    good_to_go = False
                    break
            if not good_to_go:
                continue
            page_title = page.attrs[2][1]
            links = page.findAll("a")
            link_terms = [link.text.lower() for link in links]
            link_index[page_title.lower()] = np.unique(link_terms).tolist()

    log.close()

    link_index_file = code_dir + "/data/" + "_".join(keywords) + "_hyperlinks/" + wiki_folder_arg + ".pickle"
    pickle.dump(link_index, file(link_index_file, "w+"))


def build_content_index(wiki_folder_arg, keywords, num_features=30, method="bigrams", stemming=True):
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
    if not exists(code_dir + "/data/" + "_".join(keywords) + "_" + method):
        mkdir(code_dir + "/data/" + "_".join(keywords) + "_" + method)
    log_file = code_dir + "/data/" + "_".join(keywords) + "_" + method + "/" + wiki_folder_arg + ".txt"
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
            good_to_go = True
            for keyword in keywords:
                if keyword not in plain_text:
                    good_to_go = False
                    break
            if not good_to_go:
                continue
            page_title = page.attrs[2][1]

            #Tokenize the text to words
            tokenized_text = [alphabetic_tokenizer.tokenize(s) for s in nltk.sent_tokenize(plain_text)]
            tokenized_text = np.concatenate(tokenized_text)

            #Do stemming and remove stopwords
            if stemming:
                tokenized_text = [stemmer.stem(w) for w in tokenized_text if
                                  not w in nltk.corpus.stopwords.words('english')]
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

    content_index_file = code_dir + "/data/" + "_".join(keywords) + "_" + method + "/" + wiki_folder_arg + ".pickle"
    pickle.dump(content_index, file(content_index_file, "w+"))


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


def get_page_content(page_title, wiki_index):
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
            return " ".join(page.findAll(text=True))


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


def run(all_args, target_func, func_args=(), process_limit=8):
    for i in xrange(0, len(all_args), process_limit):
        cur_args = all_args[i:i + process_limit]
        processes = []
        for folder in cur_args:
            cur_func_args = (folder, ) + func_args
            p = Process(target=target_func, args=cur_func_args)
            processes.append(p)
            p.start()

        for p in processes:
            p.join()


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