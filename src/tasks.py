import pickle
import codecs
import requests
from os.path import expanduser, exists, basename

from os import chdir
chdir(expanduser('~')+'/workspace/relation-extraction/src')
import relation_extraction as relex
from mycelery import app

#Data
#wiki_index = pickle.load(file(expanduser('~')+'/workspace/relation-extraction/data/wiki_index.pickle'))
wiki_index = {}


@app.task
def plain_text(page, f_path):
    if exists(f_path):
        print False
    p = relex.Page(page)
    p.set_content(wiki_index)
    try:
        p.clean_content()
    except:
        print False
    try:
        p.serialize_content(f_path)
    except:
        print False
    print True


@app.task
def entity_link(input_file, out_dir, spotlight_server):
    text = codecs.open(input_file).read()
    res = requests.post('http://'+spotlight_server+'.s.upf.edu:2222/rest/annotate',
                        data={'text': text, 'confidence': '0.2', 'support': '5'},
                        headers={"Accept": "application/json"})
    output_file = out_dir+basename(input_file)[:-4]+'.pickle'
    pickle.dump(res.json(), file(output_file, 'w'))
