# -*- coding: utf-8 -*-
from __future__ import division

import codecs
import numpy as np
from BeautifulSoup import BeautifulSoup
import colorama as c
import networkx as nx
from os.path import expanduser
from os import chdir

home = expanduser("~")
chdir(home+"/workspace/relation-extraction/src/")

import graph_filtering as gf
reload(gf)

data_dir = home+"/data/wiki/extracted/"


def get_summary(wiki_index, page_title):
    page_content = ""

    try:
        f = data_dir + wiki_index[page_title]
        data = codecs.open(f, 'r', 'utf-8').read()
        soup = BeautifulSoup(data)
    except KeyError:
        print "Cannot find entry for ", page_title, " in wiki_index"
        return
    except IOError:
        print "Cannot open the file at ", f
        return
    except (UnicodeDecodeError, UnicodeEncodeError):
        print "Cannot open file for ", page_title
        return
    pages = soup.findAll('doc')
    for page in pages:
        title = page.attrs[2][1]
        if title.lower() == page_title:
            page_content = " ".join(page.findAll(text=True))

    return page_content[:1000]


def annotate(wiki_index, page_title):
    page_content = get_summary(wiki_index, page_title)

    print c.Fore.CYAN + c.Style.BRIGHT + page_title.title()
    print c.Fore.YELLOW + c.Style.NORMAL + page_content

    annotation = raw_input("Carnatic? Y/n/d: ")
    if annotation == "" or annotation.lower() == "y":
        return "good"
    elif annotation.lower() == "n":
        return "bad"
    elif annotation.lower() == "d":
        return "doubtful"
    elif annotation.lower() == "x":
        return "unknown"
    else:
        print c.Fore.RED + "Annotation unrecognized, let's do it again.\n"
        return annotate(wiki_index, page_title)


def expand_categories(keyword, fraction=1.0):
    search_graph_file = home+'/workspace/relation-extraction/data/wiki_search/'+keyword+'_hyperlinks.graphml'
    cat_graph_file = home+'/workspace/relation-extraction/data/wiki_categories/'+keyword+'_hyperlinks.graphml'

    sg = nx.read_graphml(search_graph_file, node_type=unicode)
    cg = nx.read_graphml(cat_graph_file, node_type=unicode)

    pageranks = nx.pagerank(cg).items()
    pageranks = sorted(pageranks, key=lambda x: x[1], reverse=True)

    n = int(len(pageranks)*fraction)
    thresh = np.mean([i[1] for i in pageranks[:n]])

    wg = gf.WikiGraph(sg, {})

    wg.set_trusts(pageranks[:n])
    scoreindex = wg.trustrank(iterations=100, trust_thresh=thresh, initialize=False)
    wg.set_trusts(scoreindex)

    added_nodes = []
    for n in wg.graph.nodes():
        if n not in cg.nodes() and 'trust' in wg.graph[n] and wg.graph[n]['trust'] >= thresh:
            added_nodes.append(n)

    return added_nodes


def wiki_annotate(keyword, fraction_expansion=1.0):
    search_pages_file = home+'/workspace/relation-extraction/data/wiki_search/'+keyword+'_pages.txt'
    cat_pages_file = home+'/workspace/relation-extraction/data/wiki_categories/'+keyword+'_pages.txt'

    search_pages = [i.strip().lower() for i in codecs.open(search_pages_file, 'r', 'utf-8').readlines()]
    cat_pages = [i.strip().lower() for i in codecs.open(cat_pages_file, 'r', 'utf-8').readlines()]

    annotations = {}
    good_pages = set(cat_pages).intersection(search_pages)
    if fraction_expansion != 0:
        good_pages = good_pages.union(expand_categories(keyword, fraction=fraction_expansion))
    bad_pages = set(search_pages) - good_pages
    for page in good_pages:
        annotations[page] = 'good'
    for page in bad_pages:
        annotations[page] = 'bad'

    return annotations


def run(wiki_index, page_titles):
    annotations = {}
    for page_title in page_titles:
        try:
            res = annotate(wiki_index, page_title)
            if res != "unknown":
                annotations[page_title] = res
        except:
            continue
    return annotations


if __name__ == "__main__":
    print "You should call the run function instead."
