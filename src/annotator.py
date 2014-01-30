# -*- coding: utf-8 -*-
from __future__ import division
import codecs
from BeautifulSoup import BeautifulSoup
import colorama as c


data_dir = "/mnt/compmusic/users/gkoduri/data/wiki/extracted/"


def get_summary(wiki_index, page_title):
    page_content = ""

    f = data_dir + wiki_index[page_title]
    data = codecs.open(f, 'r', 'utf-8').read()
    try:
        soup = BeautifulSoup(data)
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

    print c.Fore.CYAN + c.Style.BRIGHT + page_title
    print c.Fore.YELLOW + page_content

    annotation = raw_input("Carnatic? Y/n/d: ")
    if annotation == "" or annotation.lower() == "y":
        return "carnatic"
    elif annotation.lower() == "n":
        return "non-carnatic"
    elif annotation.lower() == "d":
        return "doubtful"
    else:
        print c.Fore.RED + "Annotation unrecognized, let's do it again.\n"
        return annotate(wiki_index, page_title)


def run(wiki_index, page_titles):
    annotations = {}
    for page_title in page_titles:
        annotations[page_title] = annotate(wiki_index, page_title)
    return annotations


if __name__ == "__main__":
    print "You should call the run function instead."