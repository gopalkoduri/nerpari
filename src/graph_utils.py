import cStringIO
import graph_tool.all as gt
import networkx as nx
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
    for u, v, d in filt_g.edges(data=True):
        if weight_type == "distance":
            if d["weight"] > thresh:
                filt_g.remove_edge(u, v)
        elif weight_type == "similarity":
            if d["weight"] < thresh:
                filt_g.remove_edge(u, v)

    nodes = filt_g.nodes()
    for n in nodes:
        if filt_g.degree(n) == 0:
            filt_g.remove_node(n)

    return filt_g


def invert_weights(g):
    for u, v, d in g.edges(data=True):
        d["weight"] = 1.0-d["weight"]+0.000001
    return g
