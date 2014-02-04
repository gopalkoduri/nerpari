# -*- coding: utf-8 -*-

from __future__ import division
import graph_tool.all as gt
import networkx as nx
import community as c
import pickle
import cStringIO
import numpy as np


def propmap_len(vertices, propmap):
    count = 0
    for v in vertices:
        if propmap[v]:
            count += 1
    return count


def convert_graph(src_g, to="graphtool"):
    buf = cStringIO.StringIO()
    if to == "graphtool":
        nx.write_graphml(src_g, buf, encoding='utf-8')
        buf.flush()
        buf.reset()
        dest_g = gt.Graph()
        dest_g.load(buf, fmt="xml")
    else:
        src_g.save(buf, fmt="xml")
        buf.flush()
        buf.reset()
        dest_g = nx.read_graphml(buf, node_type=unicode)
    return dest_g


def sample_graph(g, num_nodes=100):
    nodes = np.array(g.nodes())
    indices = np.unique(np.random.random_integers(0, len(nodes), num_nodes+100))[:num_nodes]
    nodes = nodes[indices]
    sub_graph = g.subgraph(nodes)
    return sub_graph


def filter_edgeweight(g, thresh, weight_type="distance"):
    filt_g = g.copy()
    for u,v,d in filt_g.edges(data=True):
        if weight_type == "distance":
            if d["weight"] > thresh:
                filt_g.remove_edge(u,v)
        elif weight_type == "similarity":
            if d["weight"] < thresh:
                filt_g.remove_edge(u,v)

    nodes = filt_g.nodes()
    for n in nodes:
        if filt_g.degree(n) == 0:
            filt_g.remove_node(n)

    return filt_g


def invert_weights(g):
    for u,v,d in g.edges(data=True):
        d["weight"] = 1.0-d["weight"]+0.000001
    return g


def ochiai_coefficient(x, y):
    x = set(x)
    y = set(y)
    return len(x.intersection(y))/(np.sqrt(len(x)*len(y)))


def select_seedset(g, seedset_size):
    """
    Give a graph, this gets a bunch of nodes (seed set) of size seedset_size.
    The selection is based on inverse pagerank of the nodes.
    """
    rev_g = g.reverse(copy=True)
    inv_pageranks = nx.pagerank(rev_g)
    inv_pageranks = inv_pageranks.items()
    inv_pageranks = sorted(inv_pageranks, key=lambda x: x[1], reverse=True)
    selected_nodes = [i[0] for i in inv_pageranks[:seedset_size]]

    return selected_nodes


def propagate_trust(g, clean_g, src_bunch, method="basic", damp_factor=0.85, split_with="num_edges"):
    """
    Propagates trust from a given set of nodes to their neighbors.
    The function expects the properties 'trust' defined on all src_bunch.
    It returns the modified graph and those set of nodes which are assigned
    trust during this execution/iteration.
    
    Methods available:
    
    basic: It propagates the trust value of the src node to it's neighbor 
    without modification. An average of the total sum received at the 
    destination node becomes its trust score.
    
    dampening: It propagates the trust value of the src node to it's neighbor
    by dampening the trust by a given damp_factor. An average of the total sum 
    received at the destination node becomes its trust score.
    
    splitting: It propagates the trust value of the src node to it's neighbor
    by splitting the trust among the neighbors. The split can be based on the absolute
    number of out edges, or their weights (split_with argument can num_edges/weights).
    An average of the total sum received at the destination node becomes its trust score.
    
    """
    fresh_bunch = set()
    
    for src in src_bunch:
        if "trust" not in g[src].keys():
            print src, "has no trust assigned."
            continue
            
        neighbors = [i[1] for i in clean_g.out_edges(src)]
        if len(neighbors) == 0:
            print src, "does not have out going edges."
            continue
        
        fresh_bunch = fresh_bunch.union(neighbors)
        #To avoid recomputing over and again...
        if method == "splitting" and split_with == "num_edges":
            trust_share = g[src]["trust"]/len(neighbors)
        
        for dest in neighbors:
            if method == "basic":
                if "trust" in g[dest].keys():
                    g[dest]["trust"] += g[src]["trust"]
                    g[dest]["trust"] /= 2.0
                else:
                    g[dest]["trust"] = g[src]["trust"]
                        
            elif method == "dampening":
                if "trust" in g[dest].keys():
                    g[dest]["trust"] += g[src]["trust"]*damp_factor
                    g[dest]["trust"] /= 2.0
                else:
                    g[dest]["trust"] = g[src]["trust"]*damp_factor
                
            elif method == "splitting":
                if method == "splitting" and split_with == "weights":
                    trust_share = g[src]["trust"]*g[src][dest]["weight"]
                    
                if "trust" in g[dest].keys():
                    g[dest]["trust"] += trust_share
                    g[dest]["trust"] /= 2.0
                else:
                    g[dest]["trust"] = trust_share
                
            elif method == "hybrid":
                pass
            else:
                print "Method not implemented"
                return
    
    return g, fresh_bunch


def evaluate(expanded_seedset, annotations):
    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for n in expanded_seedset["good"]:
        if n in annotations.keys():
            if annotations[n] == "carnatic":
                tp += 1
            else:
                fp += 1

    for n in expanded_seedset["bad"] + expanded_seedset["orphans"]:
        if n in annotations.keys():
            if annotations[n] != "carnatic":
                tn += 1
            else:
                fn += 1

    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    f_measure = 2*tp/(2*tp+fp+fn)
    return precision, recall, f_measure


def community_filter(g, min_comsize, expanded_seedset):
    g = g.to_undirected()
    partition = c.best_partition(g)

    categories = {}

    for com in set(partition.values()):
        list_nodes = [nodes for nodes in partition.keys() if partition[nodes] == com]
        # print com, len(list_nodes)
        # print
        if len(list_nodes) > min_comsize:
            categories[com] = list_nodes

    # for com, nbunch in categories.items():
    #     print com, len(nbunch), nbunch[:100]

    good_nodes = []
    bad_nodes = []
    for com, nbunch in categories.items():
        good_oc = ochiai_coefficient(expanded_seedset["good"], nbunch)
        bad_oc = ochiai_coefficient(expanded_seedset["bad"], nbunch)

        if bad_oc != 0:
            if good_oc/bad_oc > 3:
                good_nodes.append(nbunch)
            else:
                bad_nodes.append(nbunch)
        else:
            good_nodes.append(nbunch)
    return {"good": good_nodes, "bad": bad_nodes}


def set_seedtrusts(g, seedset, annotations, doubtful="non-carnatic"):
    if doubtful == "non-carnatic":
        for src in seedset:
            if annotations[src] == "carnatic":
                g[src]["trust"] = 1
            elif annotations[src] != "carnatic":
                g[src]["trust"] = -1
    elif doubtful == "carnatic":
        for src in seedset:
            if annotations[src] != "non-carnatic":
                g[src]["trust"] = 1
            elif annotations[src] == "non-carnatic":
                g[src]["trust"] = -1
    else:
        for src in seedset:
            if annotations[src] == "carnatic":
                g[src]["trust"] = 1
            elif annotations[src] == "non-carnatic":
                g[src]["trust"] = -1
            else:
                g[src]["trust"] = 0
    return g


def expand_seedset(g, seedset, prop_depth=3, prop_method="dampening", damp_factor=0.85):
    fresh_bunch = seedset
    for i in xrange(prop_depth):
        g, fresh_bunch = propagate_trust(g, fresh_bunch, method=prop_method, damp_factor=damp_factor)
    return g


def run_config(g, annotations, seedset_size, prop_method="dampening", damp_factor=0.85, prop_depth=3,
               min_comsize=10, expand_steps=2):
    #select the seed set
    seedset = select_seedset(g, seedset_size)

    #set the trusts for initial seedset
    g = set_seedtrusts(g, seedset, doubtful="non-carnatic")

    #expand the seedset: trust propagation
    g = expand_seedset(g, seedset, prop_depth=prop_depth, prop_method=prop_method, damp_factor=damp_factor)
    good_nodes = []
    bad_nodes = []
    orphans = []
    for n in g.nodes():
        if "trust" in g[n].keys():
            if g[n]["trust"] > 0:
                good_nodes.append(n)
            else:
                bad_nodes.append(n)
        else:
            orphans.append(n)
    expanded_seedset = {"good": good_nodes, "bad": bad_nodes}

    if expand_steps == 1:
        return evaluate(expanded_seedset, annotations)

    #expand the seedset: community filter
    expanded_seedset = community_filter(g, min_comsize=min_comsize, expanded_seedset=expanded_seedset)
    return evaluate(expanded_seedset, annotations)


if __name__ == "__main__":
    #read inputs and parameters

    graph_file = "/homedtic/gkoduri/workspace/relation-extraction/data/carnatic_hyperlinks.graphml"
    annotation_file = "/homedtic/gkoduri/workspace/relation-extraction/data/annotations.pickle'"
    seedset_size = 50
    prop_method = "basic"
    prop_depth = 3
    damp_factor = 0.85
    min_comsize = 10

    g = nx.read_graphml(graph_file, node_type=unicode)
    print g.number_of_nodes(), g.number_of_edges()

    annotations = pickle.load(file(annotation_file))

    #run
    [p, r, f] = run_config(g, annotations, seedset_size, prop_method=prop_method, damp_factor=damp_factor,
                           prop_depth=prop_depth, min_comsize=min_comsize, expand_steps=2)

    # clean_g = nx_g.subgraph(fgood_nodes)
    # nx.write_graphml(clean_g, '/homedtic/gkoduri/workspace/relation-extraction/data/carnatic_hyperlinks_clean.graphml')

