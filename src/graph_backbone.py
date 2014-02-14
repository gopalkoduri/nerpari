def filter_graph_edges(G, DISPARITY_FILTER_SIGNIF_LEVEL, verbose=True, print_prefix=''):

    '''
    A large number of complex systems find a natural abstraction in the form of weighted networks whose nodes represent
    the elements of the system and the weighted edges identify the presence of an interaction and its relative strength.
    In recent years, the study of an increasing number of large-scale networks has highlighted the statistical
    heterogeneity of their interaction pattern, with degree and weight distributions that vary over many orders of
    magnitude. These features, along with the large number of elements and links, make the extraction of the truly
    relevant connections forming the network's backbone a very challenging problem. More specifically, coarse-graining
    approaches and filtering techniques come into conflict with the multiscale nature of large-scale systems. Here, we
    define a filtering method that offers a practical procedure to extract the relevant connection backbone in complex
    multiscale networks, preserving the edges that represent statistically significant deviations with respect to a
    null model for the local assignment of weights to edges. An important aspect of the method is that it does not
    belittle small-scale interactions and operates at all scales defined by the weight distribution. We apply our
    method to real-world network instances and compare the obtained results with alternative backbone
    extraction techniques. (http://www.pnas.org/content/106/16/6483.abstract)
    '''

    if verbose:
        print '%sFiltering with ' % print_prefix + str(100*(1-DISPARITY_FILTER_SIGNIF_LEVEL))+'% confidence ...',

    # FOR DIRECTED
    indegree = G.in_degree(weight=None)
    outdegree = G.out_degree(weight=None)
    instrength = G.in_degree(weight=None)
    outstrength = G.out_degree(weight=None)

    # FOR UNDIRECTED
    # degree = G.degree(weight=None)
    # strength = G.degree(weight='weight')

    edges = G.edges()
    for i, j in edges:
            # FOR DIRECTED
            pij = float(1)/float(outstrength[i])
            pji = float(1)/float(instrength[j])
            aij = (1-pij)**(outdegree[i]-1)
            aji = (1-pji)**(indegree[j]-1)
            if aij < DISPARITY_FILTER_SIGNIF_LEVEL or aji < DISPARITY_FILTER_SIGNIF_LEVEL:
                continue

            # FOR UNDIRECTED
            # pij = float(G[i][j]['weight'])/float(strength[i])
            # aij = (1-pij)**(degree[i]-1)
            # if aij < DISPARITY_FILTER_SIGNIF_LEVEL:
            #     continue

            G.remove_edge(i, j)
    nodes = G.nodes()
    for n in nodes:
        if G.degree(n) < 1:
            print n
            G.remove_node(n)
    if verbose:
        print G.number_of_nodes(), 'nodes,', G.number_of_edges(), 'edges'

    return G
