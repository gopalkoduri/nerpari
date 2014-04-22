# -*- coding: utf-8 -*-

from os.path import expanduser
import pickle
import codecs

home = expanduser("~")
extracted_wiki_data = home+"/data/wiki/extracted"

#TODO: Python imports
from wiki_tools import wiki_indexer as wi
reload(wi)

from . import text_processor as tp
reload(tp)


class Page(tp.Processor):
    """
    A page object has all the information about a given page.

    -> Clean content (cleaned by removeing braces and quotes,
    and filtering out sentences that are too short to be sentences)

    -> Parsed content from Stanford NLP tools

    -> Entity linked content from DBpedia spotlight

    -> Also contains functions to handle some I/O
    """

    def __init__(self, title, index=None, keyword='carnatic_music'):
        tp.Processor.__init__(self)
        self.title = title
        self.keyword = keyword

        if index:
            path = home + '/data/text-analysis/plain_text/' + keyword + '/' + unicode(index) + '.pickle'
            try:
                self.content = pickle.load(file(path))
            except IOError:
                print "Unable to locate the file with text content, tried to look here: ", path
            path = home + '/data/text-analysis/parsed_text/' + keyword + '/' + unicode(index) + '.pickle'
            try:
                self.parsed_content = pickle.load(file(path))
            except IOError:
                print "Unable to locate the file with text content, tried to look here: ", path

    def set_content(self, wiki_index):
        """
        Call this function if it fails to locate the file with text content.
        """
        self.content = wi.get_page_content(self.title, wiki_index)
        p = tp.Processor()
        self.content = p.clean_content(self.content)

    def serialize(self, path):
        pickle.dump(self, file(path, 'w'))

    def serialize_content(self, path):
        codecs.open(path, "w", "utf-8").write(self.content)
