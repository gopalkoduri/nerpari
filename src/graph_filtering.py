# -*- coding: utf-8 -*-

from __future__ import division
from copy import deepcopy
import networkx as nx
import community as c
import pickle
import numpy as np


def ochiai_coefficient(x, y):
    x = set(x)
    y = set(y)
    return len(x.intersection(y)) / (np.sqrt(len(x) * len(y)))


class WikiGraph():
    def __init__(self, nx_graph, annotations):
        self.clean_g = nx_graph
        self.graph = nx_graph.copy()
        self.annotations = annotations

        #trust propagation data
        self.seedset = []
        self.propset = []
        self.seedset_method = "pagerank"
        self.seedset_size = 70
        self.prop_method = "dampening"
        self.prop_depth = 3
        self.damp_factor = 0.85
        self.split_with = "num_edges"

        #community analysis data
        self.min_comsize = 10

        #result
        self.expanded_seedset = {"good": [], "bad": [], "orphans": []}

    def set_config(self, seedset_size, seedset_method, prop_method, prop_depth,
                   damp_factor, split_with, min_comsize):
        #trust propagation data
        self.seedset_size = seedset_size
        self.seedset_method = seedset_method
        self.prop_method = prop_method
        self.prop_depth = prop_depth
        self.damp_factor = damp_factor
        self.split_with = split_with

        #community analysis data
        self.min_comsize = min_comsize

    def select_seedset(self):
        """
        This method gets a bunch of nodes (seed set) of size seedset_size.
        The selection is based on inverse pagerank of the nodes.
        """
        if self.seedset_method == "pagerank":
            rev_g = self.graph.reverse(copy=True)
            inv_pageranks = nx.pagerank(rev_g)
            inv_pageranks = inv_pageranks.items()
            inv_pageranks = sorted(inv_pageranks, key=lambda x: x[1], reverse=True)
            self.seedset = [i[0] for i in inv_pageranks[:self.seedset_size]]

        elif self.seedset_method == "eig":
            centralities = nx.eigenvector_centrality(self.graph).items()
            centralities = sorted(centralities, key=lambda x: x[1], reverse=True)
            self.seedset = [i[0] for i in centralities[:self.seedset_size]]

        elif self.seedset_method == "outdegree":
            outdegree_centralities = nx.out_degree_centrality(self.graph).items()
            outdegree_centralities = sorted(outdegree_centralities, key=lambda x: x[1], reverse=True)
            self.seedset = [i[0] for i in outdegree_centralities[:self.seedset_size]]

    def propagate_trust(self):
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
        for i in xrange(self.prop_depth):
            fresh_bunch = set()

            if len(self.propset) == 0:
                self.propset = deepcopy(self.seedset)

            for src in self.propset:
                if "trust" not in self.graph[src].keys():
                    # print src, "has no trust assigned."
                    continue

                neighbors = [i[1] for i in self.clean_g.out_edges(src)]
                if len(neighbors) == 0:
                    # print src, "does not have out going edges."
                    continue

                fresh_bunch = fresh_bunch.union(neighbors)
                #To avoid recomputing over and again...
                if self.prop_method == "splitting" and self.split_with == "num_edges":
                    trust_share = self.graph[src]["trust"] / len(neighbors)

                for dest in neighbors:
                    if self.prop_method == "basic":
                        if "trust" in self.graph[dest].keys():
                            self.graph[dest]["trust"] += self.graph[src]["trust"]
                            self.graph[dest]["trust"] /= 2.0
                        else:
                            self.graph[dest]["trust"] = self.graph[src]["trust"]

                    elif self.prop_method == "dampening":
                        if "trust" in self.graph[dest].keys():
                            self.graph[dest]["trust"] += self.graph[src]["trust"] * self.damp_factor
                            self.graph[dest]["trust"] /= 2.0
                        else:
                            self.graph[dest]["trust"] = self.graph[src]["trust"] * self.damp_factor

                    elif self.prop_method == "splitting":
                        if self.prop_method == "splitting" and self.split_with == "weights":
                            trust_share = self.graph[src]["trust"] * self.graph[src][dest]["weight"]

                        if "trust" in self.graph[dest].keys():
                            self.graph[dest]["trust"] += trust_share
                            self.graph[dest]["trust"] /= 2.0
                        else:
                            self.graph[dest]["trust"] = trust_share

                    elif self.prop_method == "hybrid":
                        pass
                    else:
                        print "Method not implemented"
                        return

            good_nodes = []
            bad_nodes = []
            orphans = []
            for n in self.graph.nodes():
                if "trust" in self.graph[n].keys():
                    if self.graph[n]["trust"] > 0:
                        good_nodes.append(n)
                    else:
                        bad_nodes.append(n)
                else:
                    orphans.append(n)
            self.expanded_seedset = {"good": good_nodes, "bad": bad_nodes, "orphans": orphans}
            self.propset = fresh_bunch

    def evaluate(self):
        tp = 0
        tn = 0
        fp = 0
        fn = 0

        for n in self.expanded_seedset["good"]:
            if n in self.annotations.keys():
                if self.annotations[n] == "carnatic":
                    tp += 1
                else:
                    fp += 1

        for n in self.expanded_seedset["bad"] + self.expanded_seedset["orphans"]:
            if n in self.annotations.keys():
                if self.annotations[n] != "carnatic":
                    tn += 1
                else:
                    fn += 1

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f_measure = 2 * tp / (2 * tp + fp + fn)

        n_pos = len([i for i in self.annotations.keys() if self.annotations[i] == "carnatic"])
        return precision, recall, f_measure, len(self.expanded_seedset["good"])/n_pos

    def community_filter(self):
        graph = self.graph.to_undirected()
        partition = c.best_partition(graph)

        categories = {}

        for com in set(partition.values()):
            list_nodes = [nodes for nodes in partition.keys() if partition[nodes] == com]
            # print com, len(list_nodes)
            # print
            if len(list_nodes) > self.min_comsize:
                categories[com] = list_nodes

        # for com, nbunch in categories.items():
        #     print com, len(nbunch), nbunch[:100]

        good_nodes = []
        bad_nodes = []
        for com, nbunch in categories.items():
            good_oc = ochiai_coefficient(self.expanded_seedset["good"], nbunch)
            bad_oc = ochiai_coefficient(self.expanded_seedset["bad"], nbunch)

            if bad_oc != 0:
                if good_oc / bad_oc > 3:
                    good_nodes.extend(nbunch)
                else:
                    bad_nodes.extend(nbunch)
            else:
                good_nodes.extend(nbunch)

        self.expanded_seedset = {"good": good_nodes, "bad": bad_nodes, "orphans": []}

    def set_seedtrusts(self, doubtful="non-carnatic"):
        if doubtful == "non-carnatic":
            for src in self.seedset:
                if self.annotations[src] == "carnatic":
                    self.graph[src]["trust"] = 1
                elif self.annotations[src] != "carnatic":
                    self.graph[src]["trust"] = -1
        elif doubtful == "carnatic":
            for src in self.seedset:
                if self.annotations[src] != "non-carnatic":
                    self.graph[src]["trust"] = 1
                elif self.annotations[src] == "non-carnatic":
                    self.graph[src]["trust"] = -1
        else:
            for src in self.seedset:
                if self.annotations[src] == "carnatic":
                    self.graph[src]["trust"] = 1
                elif self.annotations[src] == "non-carnatic":
                    self.graph[src]["trust"] = -1
                else:
                    self.graph[src]["trust"] = 0
        return self.graph

    def run_config(self, expand_steps=2):
        #set configuration
        # self.set_config(seedset_size, seedset_method, prop_method, prop_depth,
        #                 damp_factor, split_with, min_comsize)

        #print configuration
        print """
        Run configuration:
        seedset size: %d,
        seedset method: %s,
        trust propagation method: %s,
        trust propagation depth: %d,
        trust propagation damp factor: %f,
        trust propagation split factor: %s,
        min. community size: %d,
        no. of expansion steps: %d
        """ % (self.seedset_size, self.seedset_method, self.prop_method, self.prop_depth,
               self.damp_factor, self.split_with, self.min_comsize, expand_steps)

        #select the seed set
        self.select_seedset()
        print "seedset selected."

        #set the trusts for initial seedset
        self.set_seedtrusts(doubtful="non-carnatic")
        print "trusts set for seedset."

        #expand the seedset: trust propagation
        self.propagate_trust()

        if expand_steps == 1:
            print "step.1 of expanding seedset (trust prop) is done."
            return self.evaluate()

        #expand the seedset: community filter
        self.community_filter()
        print "step.2 of expanding seedset (community analysis) is done."
        return self.evaluate()

    def vary_parameter(self, parameter_label, values, expand_steps=1):
        precisions = []
        recalls = []
        f_measures = []
        num_nodes = []

        #run
        for value in values:
            self.graph = self.clean_g.copy()
            setattr(self, parameter_label, value)
            [p, r, f, n] = self.run_config(expand_steps=expand_steps)
            precisions.append(p)
            recalls.append(r)
            f_measures.append(f)
            num_nodes.append(n)

        return {"x": values, "precisions": precisions, "recalls": recalls,
                "f_measures": f_measures, "num_nodes": num_nodes}


if __name__ == "__main__":
    pass
