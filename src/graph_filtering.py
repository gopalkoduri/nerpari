# -*- coding: utf-8 -*-

from __future__ import division
from copy import deepcopy
from math import ceil
from scipy import sparse
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
        self.doubtful = "bad"

        #trust propagation data
        self.seedset = {"good": [], "bad": [], "orphans": []}
        self.visited_set = set()
        self.seedset_method = "community-pagerank"
        self.seedset_size = 50
        self.seedset_type = "both"
        self.scores = {}  # holds pagerank/comm-pagerank/other values computed for use again
        self.prop_method = "dampening"
        self.prop_depth = 3
        self.damp_factor = 0.85
        self.split_with = "num_edges"

        #community analysis data
        self.min_comsize = 10
        self.min_seed_per_com = 3
        self.level = 1

        #voting
        self.vote_method = "count"
        self.vote_contributors = "in" # can be out, or both as well
        self.vote_iterations = 1
        self.vote_contribution = 0.5

        #result
        self.expanded_seedset = {"good": [], "bad": [], "orphans": []}

        #evaluation data
        self.tp_nodes = []
        self.fp_nodes = []
        self.tn_nodes = []
        self.fn_nodes = []

    def reset_data(self, scores=True):
        if scores:
            self.scores = {}

        self.graph = self.clean_g.copy()

        #trust propagation data
        self.seedset = {"good": [], "bad": [], "orphans": []}
        self.visited_set = set()

        #result
        self.expanded_seedset = {"good": [], "bad": [], "orphans": []}
        self.tp_nodes = []
        self.fp_nodes = []
        self.tn_nodes = []
        self.fn_nodes = []

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
        selected_nodes = []
        if self.seedset_method == "pagerank":
            if self.scores == {}:
                if type(self.graph) == nx.classes.digraph.DiGraph:
                    rev_g = self.graph.reverse(copy=True)
                else:
                    rev_g = self.graph
                scores = nx.pagerank(rev_g).items()
                scores = sorted(scores, key=lambda x: x[1], reverse=True)
                self.scores = scores

            selected_nodes = []
            if self.seedset_type == "both":
                for i in self.scores:
                    if i[0] in self.annotations.keys():
                        selected_nodes.append(i[0])
                    if len(selected_nodes) >= self.seedset_size:
                        break
            elif self.seedset_type == "pos":
                for i in self.scores:
                    if i[0] in self.annotations.keys() and self.annotations[i[0]] == "good":
                        selected_nodes.append(i[0])
                    if len(selected_nodes) >= self.seedset_size:
                        break
            else:
                for i in self.scores:
                    if i[0] in self.annotations.keys() and self.annotations[i[0]] != "good":
                        selected_nodes.append(i[0])
                    if len(selected_nodes) >= self.seedset_size:
                        break

        elif self.seedset_method == "eig":
            if self.scores == {}:
                scores = nx.eigenvector_centrality(self.graph).items()
                scores = sorted(scores, key=lambda x: x[1], reverse=True)
                self.scores = scores
            selected_nodes = []
            if self.seedset_type == "both":
                for i in self.scores:
                    if i[0] in self.annotations.keys():
                        selected_nodes.append(i[0])
                    if len(selected_nodes) >= self.seedset_size:
                        break
            elif self.seedset_type == "pos":
                for i in self.scores:
                    if i[0] in self.annotations.keys() and self.annotations[i[0]] == "good":
                        selected_nodes.append(i[0])
                    if len(selected_nodes) >= self.seedset_size:
                        break
            else:
                for i in self.scores:
                    if i[0] in self.annotations.keys() and self.annotations[i[0]] != "good":
                        selected_nodes.append(i[0])
                    if len(selected_nodes) >= self.seedset_size:
                        break

        elif self.seedset_method == "outdegree":
            if self.scores == {}:
                if type(self.graph) == nx.classes.digraph.DiGraph:
                    scores = nx.out_degree_centrality(self.graph).items()
                else:
                    scores = nx.degree_centrality(self.graph).items()
                scores = sorted(scores, key=lambda x: x[1], reverse=True)
                self.scores = scores
            selected_nodes = []
            if self.seedset_type == "both":
                for i in self.scores:
                    if i[0] in self.annotations.keys():
                        selected_nodes.append(i[0])
                    if len(selected_nodes) >= self.seedset_size:
                        break
            elif self.seedset_type == "pos":
                for i in self.scores:
                    if i[0] in self.annotations.keys() and self.annotations[i[0]] == "good":
                        selected_nodes.append(i[0])
                    if len(selected_nodes) >= self.seedset_size:
                        break
            else:
                for i in self.scores:
                    if i[0] in self.annotations.keys() and self.annotations[i[0]] != "good":
                        selected_nodes.append(i[0])
                    if len(selected_nodes) >= self.seedset_size:
                        break

        elif self.seedset_method == "community-pagerank" or self.seedset_method == "community-hubs":
            if self.scores == {}:
                if self.seedset_method == "community-pagerank":
                    rev_g = self.graph.reverse(copy=True)
                    scores = nx.pagerank(rev_g)
                    self.scores = scores
                else:
                    scores, authority_scores = nx.hits(self.graph)
                    self.scores = scores

            ug = self.graph.to_undirected()  # This returns a deep copy
            dgram = c.generate_dendogram(ug)
            partition = c.partition_at_level(dgram, self.level)

            categories = {}
            num_com_nodes = 0
            for com in set(partition.values()):
                com_nodes = [nodes for nodes in partition.keys() if partition[nodes] == com]
                if len(com_nodes) > self.min_comsize:
                    categories[com] = com_nodes
                    num_com_nodes += len(com_nodes)

            selected_nodes = {}
            for com, nbunch in categories.items():
                max_num_seed = int(ceil(self.seedset_size*len(nbunch)/num_com_nodes))
                if max_num_seed < self.min_seed_per_com:
                    max_num_seed = self.min_seed_per_com
                temp = {}
                for node in nbunch:
                    if self.seedset_type == "both":
                        #if node in self.annotations.keys():
                        temp[node] = self.scores[node]
                    elif self.seedset_type == "pos":
                        #if node in self.annotations.keys() and self.annotations[node] == "good":
                        if self.annotations[node] == "good":
                            temp[node] = self.scores[node]
                    elif self.seedset_type == "neg":
                        #if node in self.annotations.keys() and self.annotations[node] != "good":
                        if self.annotations[node] != "good":
                            temp[node] = self.scores[node]

                temp = temp.items()
                temp = sorted(temp, key=lambda x: x[1], reverse=True)
                for i in temp[:max_num_seed]:
                    selected_nodes[i[0]] = i[1]

            selected_nodes = sorted(selected_nodes.items(), key=lambda x: x[1], reverse=True)
            selected_nodes = [i[0] for i in selected_nodes[:self.seedset_size]]

        # return selected_nodes
        self.seedset = {"good": [], "bad": [], "orphans": []}
        for node in selected_nodes:
            if self.annotations[node] == "good":
                self.seedset["good"].append(node)
            elif self.annotations[node] == "bad":
                self.seedset["bad"].append(node)
            elif self.annotations[node] == "doubtful":
                if self.doubtful == "good":
                    self.seedset["good"].append(node)
                elif self.doubtful == "bad":
                    self.seedset["bad"].append(node)
                else:
                    self.seedset["orphans"].append(node)

    def set_seedtrusts(self, seedset="seedset"):
        seedset = getattr(self, seedset)
        for n in seedset["good"]:
            # if "trust" in self.graph[n]:
            #     continue
            self.graph[n]["trust"] = 1
        for n in seedset["bad"]:
            # if "trust" in self.graph[n]:
            #     continue
            self.graph[n]["trust"] = -1

    def set_trusts(self, scoreindex):
        for node, trust in scoreindex:
            self.graph[node]["trust"] = trust

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
        #Initialize the current_bunch to the seedset
        #NOTE: make sure the trusts are already assigned for them!

        current_bunch = np.concatenate((self.seedset["good"], self.seedset["bad"]))
        for i in xrange(self.prop_depth):
            #Current bunch is assumed to have trusts assigned
            self.visited_set = self.visited_set.union(current_bunch)

            #Next bunch is a union of all the neighbors of the current bunch
            next_bunch = set()

            for src in current_bunch:
                if "trust" not in self.graph[src].keys():
                    # print src, "has no trust assigned."
                    continue

                if type(self.clean_g) == nx.classes.digraph.DiGraph:
                    neighbors = [i[1] for i in self.clean_g.out_edges(src)]
                else:
                    neighbors = [i[1] for i in self.clean_g.edges(src)]

                #Here we take a heuristic that the first assigned score is probably
                #our best bet (unlike trustrank which iterates over and modifies it
                #over and again).

                neighbors = [dest for dest in neighbors if dest not in self.visited_set]
                if len(neighbors) == 0:
                    # print src, "does not have any new out going edges."
                    continue

                next_bunch = next_bunch.union(neighbors)

                #To avoid recomputing over and again ...
                if self.prop_method == "splitting" and self.split_with == "num_edges":
                    #Note that the definition of neighbors has changed. We are not
                    #dividing by the actual number of neighbors, but by the "new"
                    #neighbors that we haven't yet seen in the previous iterations of
                    #the propagation procedure.
                    trust_share = self.graph[src]["trust"] / len(neighbors)

                #The trust is averaged for a node if it receives trust values from more
                #than one source. We allow trust values from multiple sources in a given
                #iteration, but not across iterations.
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
                    else:
                        print "Method not implemented"
                        return

            current_bunch = next_bunch

        good_nodes = []
        bad_nodes = []
        orphans = []
        score_index = []
        for n in self.graph.nodes():
            if "trust" in self.graph[n].keys():
                score_index.append((n, self.graph[n]["trust"]))
                if self.graph[n]["trust"] > 0:
                    good_nodes.append(n)
                else:
                    bad_nodes.append(n)
            else:
                orphans.append(n)
        self.expanded_seedset = {"good": good_nodes, "bad": bad_nodes, "orphans": orphans}
        return score_index

    def evaluate(self):
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        self.tp_nodes = []
        self.fp_nodes = []
        self.tn_nodes = []
        self.fn_nodes = []

        node_labels = {}  # these represent the status of a node as per expanded seedset
        for n in self.expanded_seedset['good']:
            node_labels[n] = 'good'
        for n in self.expanded_seedset['bad']:
            node_labels[n] = 'bad'
        for n in self.expanded_seedset['orphans']:
            node_labels[n] = 'orphans'

        remaining_nodes = set(self.annotations.keys())-set(node_labels.keys())
        for n in remaining_nodes:
            node_labels[n] = 'orphans'                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          

        for n in self.annotations.keys():
            if self.doubtful == "bad":
                if node_labels[n] == "good":
                    if self.annotations[n] == "good":
                        tp += 1
                        self.tp_nodes.append(n)
                    else:
                        fp += 1
                        self.fp_nodes.append(n)
                else:
                    if self.annotations[n] != "good":
                        tn += 1
                        self.tn_nodes.append(n)
                    else:
                        fn += 1
                        self.fn_nodes.append(n)
            else:
                if node_labels[n] == "good":
                    if self.annotations[n] != "bad":
                        tp += 1
                        self.tp_nodes.append(n)
                    else:
                        fp += 1
                        self.fp_nodes.append(n)
                else:
                    if self.annotations[n] == "bad":
                        tn += 1
                        self.tn_nodes.append(n)
                    else:
                        fn += 1
                        self.fn_nodes.append(n)

        #for n in self.annotations.keys():
        #    if self.doubtful == "bad":
        #        if n in self.expanded_seedset["good"]:
        #            if self.annotations[n] == "good":
        #                tp += 1
        #                self.tp_nodes.append(n)
        #            else:
        #                fp += 1
        #                self.fp_nodes.append(n)
        #        else:
        #            if self.annotations[n] != "good":
        #                tn += 1
        #                self.tn_nodes.append(n)
        #            else:
        #                fn += 1
        #                self.fn_nodes.append(n)
        #    else:
        #        if n in self.expanded_seedset["good"]:
        #            if self.annotations[n] != "bad":
        #                tp += 1
        #                self.tp_nodes.append(n)
        #            else:
        #                fp += 1
        #                self.fp_nodes.append(n)
        #        else:
        #            if self.annotations[n] == "bad":
        #                tn += 1
        #                self.tn_nodes.append(n)
        #            else:
        #                fn += 1
        #                self.fn_nodes.append(n)

        print tp, fp, tn, fn
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f_measure = 2 * tp / (2 * tp + fp + fn)

        return precision, recall, f_measure

    def community_filter(self):
        if type(self.graph) != nx.classes.graph.Graph:
            graph = self.clean_g.to_undirected()
        else:
            graph = self.clean_g.copy()
        dendogram = c.generate_dendogram(graph)
        #partition = c.best_partition(graph)
        partition = c.partition_at_level(dendogram, level=self.level)

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
        score_index = []
        for com, nbunch in categories.items():
            good_oc = ochiai_coefficient(self.expanded_seedset["good"], nbunch)
            bad_oc = ochiai_coefficient(self.expanded_seedset["bad"], nbunch)
            #print good_oc, bad_oc, (good_oc-bad_oc)/(good_oc+bad_oc), len(nbunch)
            #print nbunch[:20]
            #print

            goodness_index = (good_oc-bad_oc)/(good_oc+bad_oc)
            for node in nbunch:
                if "trust" not in self.graph[node].keys():
                    score_index.append((node, goodness_index))

            if bad_oc != 0:
                if goodness_index > 0:
                    good_nodes.extend(nbunch)
                else:
                    bad_nodes.extend(nbunch)
            else:
                good_nodes.extend(nbunch)

        good_nodes.extend(self.expanded_seedset["good"])
        bad_nodes = list(set(bad_nodes) - set(good_nodes))
        self.expanded_seedset = {"good": good_nodes, "bad": bad_nodes, "orphans": []}
        return score_index

    def run_config(self, expand_steps=3):
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
        self.set_seedtrusts(seedset="seedset")
        print "trusts set for seedset."

        #expand the seedset: trust propagation
        score_index = self.propagate_trust()
        self.set_trusts(scoreindex=score_index)
        print "step.1 of expanding seedset (trust prop) is done."

        if expand_steps == 1:
            return self.evaluate()

        #expand the seedset: community filter
        score_index = self.community_filter()
        self.set_trusts(scoreindex=score_index)
        print "step.2 of expanding seedset (community analysis) is done."

        if expand_steps == 2:
            return self.evaluate()

        #expand the seedset: voting
        self.vote_nodes()
        # score_index = self.vote_nodes()
        # self.set_trusts(scoreindex=score_index)
        print "step.3 of expanding seedset (voting) is done."

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
        community_pagerank_reach = []
        community_hubs_reach = []

        pagerank_balance = []
        eig_balance = []
        outdegree_balance = []
        community_pagerank_balance = []
        community_hubs_balance = []

        for seedset_size in seedset_sizes:
            self.seedset_size = seedset_size

            self.reset_data()
            self.seedset_method = "pagerank"
            self.select_seedset()
            pagerank_reach.append(self.seedset_reach())
            pagerank_balance.append(self.seedset_balance())
            print "pagerank eval done."

            self.reset_data()
            self.seedset_method = "eig"
            self.select_seedset()
            eig_reach.append(self.seedset_reach())
            eig_balance.append(self.seedset_balance())
            print "eig eval done."

            self.reset_data()
            self.seedset_method = "outdegree"
            self.select_seedset()
            outdegree_reach.append(self.seedset_reach())
            outdegree_balance.append(self.seedset_balance())
            print "outdegree eval done."

            self.reset_data()
            self.seedset_method = "community-pagerank"
            self.select_seedset()
            community_pagerank_reach.append(self.seedset_reach())
            community_pagerank_balance.append(self.seedset_balance())
            print "community-pagerank eval done."

            self.reset_data()
            self.seedset_method = "community-hubs"
            self.select_seedset()
            community_hubs_reach.append(self.seedset_reach())
            community_hubs_balance.append(self.seedset_balance())
            print "community-hubs eval done."

        return {"x": seedset_sizes, "pagerank_reach": pagerank_reach, "eig_reach": eig_reach,
                "outdegree_reach": outdegree_reach, "pagerank_balance": pagerank_balance,
                "eig_balance": eig_balance, "outdegree_balance": outdegree_balance,
                "community_pagerank_reach": community_pagerank_reach,
                "community_pagerank_balance": community_pagerank_balance,
                "community_hubs_reach": community_hubs_reach, "community_hubs_balance": community_hubs_balance}

    def vote_nodes(self):
        """
        This is the final step to clean up the labels. The procedure is:
        1. For each node, get it's neighbours (both in and out edges)
        2. Note the ratio of good:bad nodes among them
        3. Label the node good, if it has more good neighbors, and the same applies for bad.

        Method can be count/score.
        """
        working_g = self.graph.copy()
        for it in xrange(self.vote_iterations):
            for node in self.graph.nodes_iter():
                if node == "trust":
                    continue

                #If the voting respects direction, then we take the inward links of a node
                #to decide whether it is good/bad. Otherwise, we consider out links as well.
                if self.vote_contributors == "in":
                    neighbors = [i[0] for i in self.graph.in_edges(node)]
                elif self.vote_contributors == "out":
                    neighbors = [i[1] for i in self.graph.out_edges(node)]
                else:
                    neighbors = [i[0] for i in self.graph.in_edges(node)]
                    neighbors.extend([i[1] for i in self.graph.out_edges(node)])

                try:
                    neighbors.remove("trust")
                except ValueError:
                    pass

                n_pos = []
                n_neg = []
                for neighbor in neighbors:
                    if "trust" not in self.graph[neighbor].keys():
                        continue
                    if self.graph[neighbor]["trust"] <= 0:
                        if self.vote_method == "count":
                            n_neg.append(1)
                        else:
                            n_neg.append(self.graph[neighbor]["trust"])
                    elif self.graph[neighbor]["trust"] > 0:
                        if self.vote_method == "count":
                            n_pos.append(1)
                        else:
                            n_pos.append(self.graph[neighbor]["trust"])

                #Irrespective of the method we use for deciding if a node is good/bad,
                #we set it's trust to the average of the positiv/negative neighbors as appropriate.
                if "trust" in working_g[node].keys():
                    score = self.vote_contribution*np.average(n_pos + n_neg) + \
                            (1-self.vote_contribution)*working_g[node]["trust"]
                    if score > 0:
                        working_g[node]["trust"] = self.vote_contribution*np.average(n_pos) + \
                                                   (1-self.vote_contribution)*working_g[node]["trust"]
                    else:
                        working_g[node]["trust"] = self.vote_contribution*np.average(n_neg) + \
                                                   (1-self.vote_contribution)*working_g[node]["trust"]
                else:
                    try:
                        score = np.average(n_pos + n_neg)
                    except:
                        print node, n_pos, n_neg
                    if score > 0:
                        # TODO: hardcoded, change the damp parameter to a variable.
                        working_g[node]["trust"] = 0.85*np.average(n_pos)
                    else:
                        working_g[node]["trust"] = 0.85*np.average(n_neg)

            self.graph = working_g.copy()

        self.expanded_seedset["good"] = []
        self.expanded_seedset["bad"] = []
        self.expanded_seedset["orphans"] = []
        # score_index = []

        for node in self.graph.nodes_iter():
            if "trust" not in self.graph[node].keys():
                self.expanded_seedset["orphans"].append(node)
                continue

            # score_index.append((node, self.graph[node]["trust"]))
            if self.graph[node]["trust"] > 0:
                self.expanded_seedset["good"].append(node)
            elif self.graph[node]["trust"] <= 0:
                self.expanded_seedset["bad"].append(node)
        # return score_index

    def trustrank(self, alpha=0.85, iterations=10, trust_thresh=0, initialize=True):
        node_index = dict(enumerate(self.clean_g.nodes()))
        node_index_reverse = {}
        for k, v in node_index.items():
            node_index_reverse[v] = k
        node_list = [node_index[i] for i in xrange(len(node_index))]

        if initialize:
            trust_scores = []
            for i in xrange(len(node_index)):
                if node_index[i] in self.seedset["good"]:
                    trust_scores.append(1)
                # FOR DISTRUST
                elif node_index[i] in self.seedset["bad"]:
                    trust_scores.append(-1)
                else:
                    trust_scores.append(0)
            trust_scores = np.array(trust_scores, dtype=float)

            if sum(trust_scores == 1) != 0:
                trust_scores[trust_scores == 1] = 1.0/sum(trust_scores == 1)
            # FOR DISTRUST
            if sum(trust_scores == -1) != 0:
                trust_scores[trust_scores == -1] = -1.0/sum(trust_scores == -1)
        else:
            trust_scores = []
            for n in node_list:
                if "trust" in self.graph[n].keys():
                    trust_scores.append(self.graph[n]["trust"])
                else:
                    trust_scores.append(0)

        rows = []
        cols = []
        data = []

        for src in node_list:
            out_neighbors = [i[1] for i in self.clean_g.out_edges(src)]
            for dest in out_neighbors:
                dest_ind = node_index_reverse[dest]
                src_ind = node_index_reverse[src]
                rows.append(dest_ind)
                cols.append(src_ind)
                data.append(1.0/len(out_neighbors))

        transition_matrix = sparse.coo_matrix((data, (rows, cols)), shape=(len(node_index), len(node_index)))

        trust_scores = np.array(trust_scores).reshape(len(trust_scores), 1)
        trust_scores = sparse.csr_matrix(trust_scores, shape=trust_scores.shape, dtype=float)
        initial_scores = deepcopy(trust_scores)
        for i in xrange(iterations):
            trust_scores = alpha*transition_matrix*trust_scores + (1-alpha)*initial_scores
            trust_scores = trust_scores.todense()
            trust_scores = np.nan_to_num(trust_scores)
            trust_scores = sparse.csr_matrix(trust_scores, shape=trust_scores.shape, dtype=float)
        trust_scores = trust_scores.todense()

        #return trust_scores
        good_nodes = []
        bad_nodes = []
        orphans = []
        score_index = []

        for i in xrange(len(trust_scores)):
            score_index.append((node_index[i], trust_scores[i, 0]))
            if trust_scores[i, 0] > trust_thresh:
                good_nodes.append(node_index[i])
            else:
                bad_nodes.append(node_index[i])
        self.expanded_seedset = {"good": good_nodes, "bad": bad_nodes, "orphans": orphans}
        return score_index

if __name__ == "__main__":
    pass
