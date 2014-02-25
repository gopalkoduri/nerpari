f = 'jazz_music_token_index.pickle'
ti = pickle.load(file(f))
thresholds = [0.2, 0.3, 0.4, 0.5]
neighbors = 50

for threshold in thresholds:
    g = wg.graph_lsa(ti, num_topics=200, num_neighbors=neighbors, sim_thresh=threshold)
    out_file = f[:-18] + 'lsa_200_' + str(neighbors) + '_' + str(threshold) + '.graphml'
    nx.write_graphml(g, out_file, encoding='utf-8')
    print out_file
