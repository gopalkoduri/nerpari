# -*- coding: utf-8 -*-

from __future__ import division
from copy import deepcopy
import networkx as nx
import community as c
import numpy as np


def ochiai_coefficient(x, y):
    x = set(x)
    y = set(y)
    return len(x.intersection(y)) / (np.sqrt(len(x) * len(y)))


class WikiGraph():
    def __init__(self, nx_graph, annotations):
        self.clean_g = nx_graph
        self.graph = self.clean_g.copy()
        self.annotations = annotations

        #trust propagation data
        self.seedset = {"good": [], "bad": [], "orphans": []}
        self.visited_set = set()
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

    def reset_data(self):
        self.graph = self.clean_g.copy()

        #trust propagation data
        self.seedset = {"good": [], "bad": [], "orphans": []}
        self.visited_set = set()

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

    def select_seedset(self, doubtful="non-carnatic"):
        """
        This method gets a bunch of nodes (seed set) of size seedset_size.
        The selection is based on inverse pagerank of the nodes.
        """
        selected_nodes = []
        if self.seedset_method == "pagerank":
            if type(self.graph) == nx.classes.digraph.DiGraph:
                rev_g = self.graph.reverse(copy=True)
            else:
                rev_g = self.graph
            inv_pageranks = nx.pagerank(rev_g).items()
            inv_pageranks = sorted(inv_pageranks, key=lambda x: x[1], reverse=True)
            selected_nodes = [i[0] for i in inv_pageranks[:self.seedset_size]]

        elif self.seedset_method == "eig":
            centralities = nx.eigenvector_centrality(self.graph).items()
            centralities = sorted(centralities, key=lambda x: x[1], reverse=True)
            selected_nodes = [i[0] for i in centralities[:self.seedset_size]]

        elif self.seedset_method == "outdegree":
            if type(self.graph) == nx.classes.digraph.DiGraph:
                outdegree_centralities = nx.out_degree_centrality(self.graph).items()
            else:
                outdegree_centralities = nx.degree_centrality(self.graph).items()
            outdegree_centralities = sorted(outdegree_centralities, key=lambda x: x[1], reverse=True)
            selected_nodes = [i[0] for i in outdegree_centralities[:self.seedset_size]]

        self.seedset = {"good": [], "bad": [], "orphans": []}
        for n in selected_nodes:
            if n not in self.annotations.keys():
                print n, "not in annotations"
                continue

            if self.annotations[n] == "carnatic":
                self.seedset["good"].append(n)
            elif self.annotations[n] == "non-carnatic":
                self.seedset["bad"].append(n)
            elif self.annotations[n] == "doubtful":
                if doubtful == "carnatic":
                    self.seedset["good"].append(n)
                elif doubtful == "non-carnatic":
                    self.seedset["bad"].append(n)
                else:
                    self.seedset["orphans"].append(n)

    def set_trusts(self, seedset="seedset"):
        seedset = getattr(self, seedset)
        for n in seedset["good"]:
            if "trust" in self.graph[n]:
                continue
            self.graph[n]["trust"] = 1
        for n in seedset["bad"]:
            if "trust" in self.graph[n]:
                continue
            self.graph[n]["trust"] = -1
        # for n in seedset["orphans"]:
        #     self.graph[n]["trust"] = 0

    def seedset_reach(self):
        """
        This method computes the ratio of nodes rechable in self.prop_depth steps from all the
        self.seedset nodes, to the total number of nodes in the graph.
        """
        reach = set(np.concatenate((self.seedset["good"], self.seedset["bad"])))
        nbunch = np.concatenate((self.seedset["good"], self.seedset["bad"]))
        for i in xrange(self.prop_depth):
            if type(self.graph) == nx.classes.digraph.DiGraph:
                neighbors = [i[1] for i in self.graph.out_edges(nbunch)]
            else:
                neighbors = [i[1] for i in self.graph.edges(nbunch)]
            reach = reach.union(neighbors)
            nbunch = neighbors

        return len(reach)/self.graph.number_of_nodes()

    def seedset_balance(self):
        n_pos = len(self.seedset["good"])
        n_neg = len(self.seedset["bad"])
        return 1.0-(abs(n_pos-n_neg)/len(self.seedset["good"] + self.seedset["bad"]))

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
        current_bunch = np.concatenate((self.seedset["good"], self.seedset["bad"]))
        for i in xrange(self.prop_depth):
            next_bunch = set()

            for src in current_bunch:
                if "trust" not in self.graph[src].keys():
                    # print src, "has no trust assigned."
                    continue

                if type(self.clean_g) == nx.classes.digraph.DiGraph:
                    neighbors = [i[1] for i in self.clean_g.out_edges(src)]
                else:
                    neighbors = [i[1] for i in self.clean_g.edges(src)]
                neighbors = [dest for dest in neighbors if dest not in self.visited_set]
                if len(neighbors) == 0:
                    # print src, "does not have out going edges."
                    continue

                next_bunch = next_bunch.union(neighbors)

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
                        if self.split_with == "weights":
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
            self.visited_set = self.visited_set.union(current_bunch)
            current_bunch = next_bunch

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

        return precision, recall, f_measure

    def community_filter(self):
        if type(self.graph) != nx.classes.graph.Graph:
            graph = self.clean_g.to_undirected()
        else:
            graph = self.clean_g.copy()
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
            print good_oc, bad_oc, (good_oc-bad_oc)/(good_oc+bad_oc), com, len(nbunch)

            if bad_oc != 0:
                if (good_oc-bad_oc)/(good_oc+bad_oc) > 0.3:
                    good_nodes.extend(nbunch)
                else:
                    bad_nodes.extend(nbunch)
            else:
                good_nodes.extend(nbunch)

        self.expanded_seedset = {"good": good_nodes, "bad": bad_nodes, "orphans": []}

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
        self.select_seedset(doubtful="non-carnatic")
        print "seedset selected."

        #set the trusts for initial seedset
        self.set_trusts(seedset="seedset")
        print "trusts set for seedset."

        #expand the seedset: trust propagation
        self.propagate_trust()
        self.set_trusts(seedset="expanded_seedset")
        print "step.1 of expanding seedset (trust prop) is done."

        if expand_steps == 1:
            return self.evaluate()

        #expand the seedset: community filter
        self.community_filter()
        self.set_trusts(seedset="expanded_seedset")
        print "step.2 of expanding seedset (community analysis) is done."

        return self.evaluate()

    def vary_parameter(self, parameter_label, values, expand_steps=1):
        precisions = []
        recalls = []
        f_measures = []

        #run
        for value in values:
            self.reset_data()
            setattr(self, parameter_label, value)
            [p, r, f] = self.run_config(expand_steps=expand_steps)
            precisions.append(p)
            recalls.append(r)
            f_measures.append(f)

        return {"x": values, "precisions": precisions, "recalls": recalls,
                "f_measures": f_measures}

    def evaluate_seedset_methods(self, seedset_sizes=np.arange(20, 200, 20)):
        pagerank_reach = []
        eig_reach = []
        outdegree_reach = []

        pagerank_balance = []
        eig_balance = []
        outdegree_balance = []

        for seedset_size in seedset_sizes:
            self.seedset_size = seedset_size

            self.reset_data()
            self.seedset_method = "pagerank"
            self.select_seedset()
            pagerank_reach.append(self.seedset_reach())
            pagerank_balance.append(self.seedset_balance())

            self.reset_data()
            self.seedset_method = "eig"
            self.select_seedset()
            eig_reach.append(self.seedset_reach())
            eig_balance.append(self.seedset_balance())

            self.reset_data()
            self.seedset_method = "outdegree"
            self.select_seedset()
            outdegree_reach.append(self.seedset_reach())
            outdegree_balance.append(self.seedset_balance())

        return {"x": seedset_sizes, "pagerank_reach": pagerank_reach, "eig_reach": eig_reach,
                "outdegree_reach": outdegree_reach, "pagerank_balance": pagerank_balance,
                "eig_balance": eig_balance, "outdegree_balance": outdegree_balance}

    def vote_nodes(self, iterations=1, method="count"):
        """
        This is the final step to clean up the labels. The procedure is:
        1. For each node, get it's neighbours (both in and out edges)
        2. Note the ratio of good:bad nodes among them
        3. Label the node good, if it has more good neighbors, and the same applies for bad.

        Method can be count/score.
        """
        sub_g = self.graph.subgraph(self.expanded_seedset["good"])
        sub_g = sub_g.to_undirected()

        for node in sub_g.nodes_iter():
            if "trust" in self.graph[node].keys():
                sub_g[node]["trust"] = self.graph[node]["trust"]

        working_g = sub_g.copy()
        for it in xrange(iterations):
            for node in sub_g.nodes_iter():
                if node == "trust":
                    continue
                # print node
                neighbors = sub_g.neighbors(node)
                try:
                    neighbors.remove("trust")
                except ValueError:
                    pass

                n_pos = 0
                n_neg = 0
                for neighbor in neighbors:
                    if "trust" not in sub_g[neighbor].keys():
                        continue
                    # print "\t", neighbor
                    if sub_g[neighbor]["trust"] <= 0:
                        if method == "count":
                            n_neg += 1
                        else:
                            n_neg += sub_g[neighbor]["trust"]
                    elif sub_g[neighbor]["trust"] > 0:
                        if method == "count":
                            n_pos += 1
                        else:
                            n_pos += sub_g[neighbor]["trust"]

                if n_pos > n_neg:
                    if method == "count":
                        working_g[node]["trust"] = 1
                    else:
                        working_g[node]["trust"] = n_pos-n_neg
                elif n_neg > n_pos:
                    if method == "count":
                        working_g[node]["trust"] = -1
                    else:
                        working_g[node]["trust"] = n_neg-n_pos
            sub_g = working_g.copy()

        self.expanded_seedset["good"] = []
        self.expanded_seedset["bad"] = []
        self.expanded_seedset["orphans"] = []

        for node in sub_g.nodes_iter():
            if "trust" not in sub_g[node].keys():
                self.expanded_seedset["orphans"].append(node)
                continue
            if sub_g[node]["trust"] > 0:
                self.expanded_seedset["good"].append(node)
            elif sub_g[node]["trust"] <= 0:
                self.expanded_seedset["bad"].append(node)

if __name__ == "__main__":
    pass
