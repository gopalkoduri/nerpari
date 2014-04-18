from __future__ import division
import networkx as nx


def get_parsed(sem_out):
    #any relation is of type Head.Child.n (X , Y)
    #special cases being of type OPERATION(X)

    parsed = []
    special = {}
    data = sem_out.strip('[]\n')+', '
    data = data.split("), ")[:-1]
    for part in data:
        #part = 'raga.in.2(7:e , 10:x'
        temp = part.strip("'").split('(')

        HCn = temp[0].split('.')
        #HCn = 'raga.in.2'

        arg_info = temp[1]
        #arg_info = '7:e , 10:x'

        temp = arg_info.split(' , ')
        if len(temp) < 2:
            if HCn[0] in special.keys():
                special[HCn[0]].append(temp[0])
            else:
                special[HCn[0]] = temp #temp is a list
        else:
            X = temp[0]
            #X = '7:e'

            Y = temp[1]
            #Y = '10:x'
            temp = {'X': X, 'Y': Y, 'HCn': HCn}
            parsed.append(temp)
    return [parsed, special]


def graph_relations(parsed, special):
    rg = nx.DiGraph()
    for i in parsed:
        if i['X'][-1] == 'e':
            if len(i['HCn']) == 1:
                print 'Unhandled:', i

            elif len(i['HCn']) == 2:
                #same as when len(i['HCn']) == 3 and i['HCn'][0] == i['HCn'][1]
                #X-Y
                if i['HCn'][1] == '1':
                    edge_label = 'subject'
                else:
                    edge_label = 'object'
                rg.add_edge(i['X'], i['Y'], {'label': edge_label})

                #X->H
                rg.add_edge(i['X'], i['HCn'][0], {'label': 'predicate'})

            elif len(i['HCn']) == 3:
                #two cases: eg: H.H.2 (X, Y) and H.C.2 (X, Y)
                #first case: add X->Y with label subject (1) or object (2)
                #            add X->H with label predicate
                #
                #second case: add X->H with label predicate
                #             add H->C with label preposition
                #             add C->Y with label value

                #first case
                if i['HCn'][0] == i['HCn'][1]:
                    #X-Y
                    if i['HCn'][2] == '1':
                        edge_label = 'subject'
                    else:
                        edge_label = 'object'
                    rg.add_edge(i['X'], i['Y'], {'label': edge_label})

                    #X->H
                    if 'NEGATION' in special.keys() and i['X'] in special['NEGATION']:
                        rg.add_edge(i['X'], 'not '+i['HCn'][0], {'label': 'predicate'})
                    else:
                        rg.add_edge(i['X'], i['HCn'][0], {'label': 'predicate'})

                #second case
                else:
                    if 'NEGATION' in special.keys() and i['X'] in special['NEGATION']:
                        rg.add_edge(i['X'], 'not '+i['HCn'][0], {'label': 'predicate'})
                        rg.add_edge(i['HCn'][0], 'not '+i['HCn'][1], {'label': 'preposition', 'rel': i['X']})
                    else:
                        rg.add_edge(i['X'], i['HCn'][0], {'label': 'predicate'})
                        rg.add_edge(i['HCn'][0], i['HCn'][1], {'label': 'preposition', 'rel': i['X']})

                    if i['HCn'][2] == '2':
                        rg.add_edge(i['HCn'][1], i['Y'], {'label': 'value', 'rel': i['X']})
                    else:
                        rg.add_edge(i['X'], i['Y'], {'label': 'subject'})

        elif i['X'][-1] == 's':
            #skip if X and Y have same index number
            x_ind = i['X'].split(':')[0]
            y_ind = i['Y'].split(':')[0]
            if x_ind == y_ind:
                weight = 0.5
            else:
                weight = 1

            if len(i['HCn']) == 1:
                if i['Y'] != i['X']:
                    if 'NEGATION' in special.keys() and i['X'] in special['NEGATION']:
                        rg.add_edge(i['Y'], i['HCn'][0], {'label': 'is not a', 'weight': weight})
                    else:
                        rg.add_edge(i['Y'], i['HCn'][0], {'label': 'is a', 'weight': weight})

            elif len(i['HCn']) == 2:
                if i['HCn'][1] == '1':
                    rg.add_edge(i['Y'], i['HCn'][0], {'label': 'prefix'})
                else:
                    rg.add_edge(i['Y'], i['HCn'][0], {'label': 'suffix'})

            elif len(i['HCn']) == 3:
                if i['HCn'][0] == i['HCn'][1]:
                    print 'Unhandled', i
                else:
                    if 'NEGATION' in special.keys() and i['X'] in special['NEGATION']:
                        rg.add_edge(i['Y'], i['HCn'][1]+' '+i['HCn'][0], {'label': 'is not a', 'weight': weight})
                    else:
                        rg.add_edge(i['Y'], i['HCn'][1]+' '+i['HCn'][0], {'label': 'is a', 'weight': weight})

                    rg.add_edge(i['HCn'][1], i['HCn'][0], {'label': 'a type of', 'weight': weight})
    return rg


def get_graph(data, n, draw=False):
    print data[n]
    parsed, special = get_parsed(data[n+1])
    print parsed, special

    rg = graph_relations(parsed, special)
    if draw:
        pos = nx.graphviz_layout(rg)
        nx.draw(rg, pos)
        edge_labels=dict([((u,v,),d['label'])
                     for u,v,d in rg.edges(data=True)])
        nx.draw_networkx_edge_labels(rg, pos, edge_labels=edge_labels)
    return rg
