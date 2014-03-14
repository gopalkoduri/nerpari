# -*- coding: utf-8 -*-

from os.path import expanduser
from os import chdir
from unidecode import unidecode
import jsonrpclib
import json
import pickle
import codecs
import nltk
import re

home = expanduser("~")
data_dir = home+"/data/wiki/extracted"
code_dir = home+"/workspace/relation-extraction"

chdir(code_dir+'/src')
import wiki_indexer as wi
reload(wi)


class StanfordNLP:
    def __init__(self, port_number=8888):
        self.server = jsonrpclib.Server("http://guqin.s.upf.edu:%d" % port_number)

    def parse(self, text):
        return json.loads(self.server.parse(text))


class Page():
    def __init__(self, title):
        self.title = title
        self.content = ""
        self.parse_output = ""
        self.resolved_content = ""

        #sentence filtering settings
        self.min_words = 2
        self.min_chars = 50

        #Coreference filtering
        self.ref_max_len = 50

        self.resolved_sentences = {}
        self.raw_sentences = []

    def set_content(self, wiki_index):
        self.content = wi.get_page_content(self.title, wiki_index)

    def serialize(self, path):
        pickle.dump(self, file(path, 'w'))

    def serialize_content(self, path):
        codecs.open(path, "w", "utf-8").write(self.content)

    def set_raw_sentences(self):
        for sentence in self.parse_output['sentences']:
            self.raw_sentences.append(" ".join([i[0] for i in sentence['words']]))

    def clean_content(self):
        if self.content == "":
            return
        #Remove braces and quotes
        braces = re.findall('\(.*?\)', self.content)
        braces.extend(re.findall('\[.*?\]', self.content))
        for b in braces:
            self.content = self.content.replace(b, '')
        self.content = self.content.replace('"', '')
        self.content = self.content.replace("'", '')

        #Discard the lines too short
        lines = self.content.split('\n')
        clean_lines = []
        for line in lines:
            line = line.strip()
            if len(line) > self.min_chars and len(line.split(' ')) > self.min_words:
                clean_lines.append(line.strip('.'))

        self.content = unidecode('. '.join(clean_lines))

    def tokenize_content(self):
        """
        This function SHOULD NOT be called. It is here just in case there must be a need in future.
        """
        print "StanfordNLP takes care of tokenization as well. Don't call this function (i.e., tokenize_content())!"
        sent_tokenizer = nltk.punkt.PunktSentenceTokenizer()
        nltk.punkt.PunktTrainer(self.content, verbose=False)
        self.resolved_sentences = sent_tokenizer.tokenize(self.content)

    def parse(self):
        nlp = StanfordNLP()
        self.parse_output = nlp.parse(self.content.encode('utf-8'))

    def anaphora_resolution(self):
        self.resolved_sentences = {}
        #TODO: get the notebook stuff here
        for s in xrange(len(self.parse_output['sentences'])):
            if s not in self.resolved_sentences.keys():
                self.resolved_sentences[s] = " ".join([i[0] for i in self.parse_output['sentences'][s]['words']])
            # elif type(self.sentences[s]) == list:
            #     self.sentences[s] = " ".join(self.sentences[s])

    def filter_sentences(self):
        """
        This function SHOULD NOT be called. It is here just in case there must be a need in future.
        """
        print "filter_sentences() is a donga boochi!! It steals many good sentences, jagratta!"
        #If there is a newline in the content, split the line and keep the longest part
        for i in xrange(len(self.resolved_sentences)):
            self.resolved_sentences[i] = self.resolved_sentences[i].strip()
            if "\n" in self.resolved_sentences[i]:
                print self.resolved_sentences[i]
                parts = [(part, len(part)) for part in self.resolved_sentences[i].split("\n")]
                parts = sorted(parts, key=lambda x: x[1], reverse=True)
                self.resolved_sentences[i] = parts[0][0]

        #Filter the very short sentences
        self.resolved_sentences = [sentence for sentence in self.resolved_sentences if len(sentence) > self.min_chars
                                                                     and len(sentence.split()) > self.min_words]
































