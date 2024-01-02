import random
import sys

import matplotlib.pyplot as plt
import networkx as nx

# parameter
node_num = 30


# randomly initialize the node weights range in [0, N-1]
def node_weights_init(G):
    #weights = {i: w for i, w in enumerate(random.sample(range(node_num), node_num))} # random and unquie (old one)
    weights = {i: i+1 for i in range(node_num)} # node i has weight i
    nx.set_node_attributes(G, weights, name = 'weights')
    
# priority function for weighted MIS game    
def node_priority_init(G):
    priority = {}
    for i in range(node_num):
        priority[i] = G.nodes[i]['weights'] / (len(list(G.neighbors(i))) + 1) # the first formula on the slide
    nx.set_node_attributes(G, priority, name = 'priority')

# randomly initialize the node strategies range in [0(out), 1(in)] (for requirements 1 only)
def node_strategy_init(G):
    strategy = {}
    for i in range(node_num):
        strategy[i] = random.randint(0, 1)
    nx.set_node_attributes(G, strategy, name = 'strategy')

# randomly initialize the node strategies (N_i or null) (for requirements 2 only)
def node_strategy_init2(G):
    strategy = {}
    for i in range(node_num):
        profile = [n for n in G.neighbors(i)] + [-1] # open neighbors + unmatched
        strategy[i] = random.choice(profile)
    nx.set_node_attributes(G, strategy, name= 'strategy')

# randomly pick up one player who can improve its utilty (weighted MIS game)
def node_choose1(G):
    players = []
    for i in range(node_num): # node i
        best_response = 1 # new best response to player i
        for j in G.neighbors(i): # neighbor node j
            if G.nodes[j]['priority'] >= G.nodes[i]['priority'] and G.nodes[j]['strategy'] == 1: # pj belogns to Li and cj = 1
                best_response = 0
                break
        if G.nodes[i]['strategy'] != best_response: # if best response need to be updated
            players.append(i)
    if len(players) == 0: # reach NE
        return -1
    return random.choice(players) # randomly pick up one node

# randomly pick up one player who can improve its utilty (symmetric MDS-based IDS game)
def node_choose2(G):
    players = []
    for i in range(node_num):
        #check domination
        has_dominate = True
        M_i = [node for node in G.neighbors(i)] + [i] # closed neighbor
        for i_neighbor in M_i:
            v_j = 0
            for i_neighbor_neighbor in G.neighbors(i_neighbor):
                v_j += G.nodes[i_neighbor_neighbor]['strategy']
            if v_j == 0:
                has_dominate = False
                break
        
        #check independence
        not_independence = False
        for neighbor_node in G.neighbors(i):
           if G.nodes[neighbor_node]['strategy'] == 1:
                not_independence = True
                break


        best_response = 1
        if not_independence or has_dominate:
            best_response = 0
        if G.nodes[i]['strategy'] != best_response:
            players.append(i)
    if len(players) == 0:
        return -1
    return random.choice(players)

# randomly pick up one player who can improve its utilty (matching game)
# return node idx and strategy
def node_choose3(G):
    players_strategies = {}
    for i in range(node_num):
        point_to = G.nodes[i]['strategy'] # node i point to
        point_me = []  # nodes point to node i
        best_response = -1
        Ni_choose_null = [] # any of the neighbors choose null

        if point_to != -1 and i == G.nodes[point_to]['strategy']: # if pair matched, then do nothing
            continue
        
        for ni in G.neighbors(i):
            if G.nodes[ni]['strategy'] == i:
                point_me.append(ni)
            elif G.nodes[ni]['strategy'] == -1:
                Ni_choose_null.append(ni)

        if len(point_me) != 0: # if pair not matched yet and some neighbors point me, then randomly choose one
            best_response = random.choice(point_me)
        elif len(Ni_choose_null) != 0: # if pair not matched yet and no neighbor point me, then choose last one
            best_response = random.choice(Ni_choose_null)

        if best_response != point_to:
            players_strategies[i] = best_response
    if len(players_strategies) == 0:
        return -1, -1
    return random.choice(list(players_strategies.items()))

           
# initialize the graph based on input
def graph_init(node_num, relations):
    G = nx.Graph()
    for i in range(0, node_num):
        G.add_node(i)
    for i, relation in enumerate(relations):
        for j, r in enumerate(relation):
            if r == '1':
               G.add_edge(i, j)
    return G

def weighted_MIS_game(graph):
    max_val = 0
    max_G = graph
    total_move_count = 0
    total_set_cardinality = 0
    for i in range(1000): # play 1000 times and observe the max cardinality value
        # initialize the graph and plot it
        G = graph
        node_weights_init(G)
        node_priority_init(G)
        node_strategy_init(G)

        move_count = 0
        node_count = 0
        while True:
            player_i = node_choose1(G)
            if player_i == -1:
                break
            G.nodes[player_i]['strategy'] = abs(1 - G.nodes[player_i]['strategy'])
            move_count += 1

        for n in range(node_num):
            if G.nodes[n]['strategy'] == 1:
                node_count += 1
        total_move_count += move_count
        total_set_cardinality += node_count
        if node_count > max_val:
            max_val = node_count
            max_G = G.copy()

    print("the cardinality of Weighted MIS Game is ", max_val)

    # plot result
    # max_G = nx.relabel_nodes(max_G, lambda x: x +1) # map the index from 0 ~ n-1 to 1 ~ n
    # pos = nx.circular_layout(max_G)
    # color = ["yellow" if max_G.nodes[i]['strategy'] == 1 else "gray" for i in max_G.nodes]
    # label = {node:str(node)+"_"+str(max_G.nodes[node]['weights']) for node in max_G.nodes}
    # nx.draw(max_G, pos, node_color = color, labels = label)
    # plt.savefig("g1.jpg") # plot last time graph result
    # plt.clf() # clear figure

def symmetric_MDS_based_IDS_game(graph):
    min_val = len(graph.nodes())
    min_G = graph
    total_move_count = 0
    total_set_cardinality = 0

    for i in range(80000):
        move_count = 0
        node_count = 0

        G = graph
        node_strategy_init(G)
        while True:
            player_i = node_choose2(G)
            if player_i == -1:
                break
            G.nodes[player_i]['strategy'] = abs(1 - G.nodes[player_i]['strategy'])
            move_count += 1

        for n in range(node_num):
            if G.nodes[n]['strategy'] == 1:
                node_count += 1
        total_move_count += move_count
        total_set_cardinality += node_count
        if node_count < min_val:
            min_val = node_count
            min_G = G.copy()

    print("the cardinality of Symmetric MDS-based IDS Game is", min_val)

    # plot result
    # min_G = nx.relabel_nodes(min_G, lambda x: x +1)
    # pos = nx.circular_layout(min_G)
    # color = ["yellow" if min_G.nodes[i]['strategy'] == 1 else "gray" for i in min_G.nodes]
    # nx.draw(min_G, pos, node_color = color, with_labels = True)
    # plt.savefig("g2.jpg") # plot last time graph result

def maximal_matching_game(graph):
    max_val = 0
    max_G = graph
    total_move_count = 0
    total_set_cardinality = 0
    for i in range(1000):
        move_count = 0
        match_edges = 0

        G = graph
        node_strategy_init2(G)
        while True:
            player_i, strategy = node_choose3(G)
            if player_i == -1:
                break
            G.nodes[player_i]['strategy'] = strategy
            move_count += 1
        total_move_count += move_count
        for n in range(node_num): # count how many edges
            if G.nodes[n]['strategy'] != -1 and G.nodes[G.nodes[n]['strategy']]['strategy'] == n:
                match_edges += 1
        match_edges /= 2 # divided by 2
        total_set_cardinality += match_edges
        if match_edges > max_val:
            max_val = match_edges
            max_G = G.copy()
    
    edge_color = []
    for n1, n2 in max_G.edges:
        if max_G.nodes[n1]['strategy'] == n2 and max_G.nodes[n2]['strategy'] == n1:
            edge_color.append("yellow")
        else:
            edge_color.append("black")
    print("the cardinality of Matching Game is", int(max_val))

    # plot result
    # max_G = nx.relabel_nodes(max_G, lambda x: x +1)
    # pos = nx.circular_layout(max_G)
    # nx.draw(max_G, pos, node_color = ["gray"], edge_color = edge_color, with_labels = True)
    #plt.savefig("g3.jpg") # plot last time graph result

if __name__ == '__main__':
    # input parameters
    node_num = int(sys.argv[1])
    params = sys.argv[2:]
    G = graph_init(node_num, params)

    # 1-1: weighted MIS game
    print("Requirement 1-1:")
    G1_1 = G.copy()
    weighted_MIS_game(G1_1)


    # 1-2: Symmetric MDS-based IDS Game
    print("Requirement 1-2:")
    G1_2 = G.copy()
    symmetric_MDS_based_IDS_game(G1_2)

    # 2: Matching Game
    print("Requirement 2:")
    G2 = G.copy()
    maximal_matching_game(G2)
    
# Ex:    
# python3 312552024_code.py 6 010000 101100 010010 010010 001101 000010
# python3 312552024_code.py 7 0110000 1010000 1101100 0010000 0010011 0000101 0000110
# python3 312552024_code.py 30 010000000000000000000000000000 101000000000000000000000000000 010100000000000000000000000000 001010000000000000000000000000 000101000000000000000000000000 000010100000000000000000000000 000001010000000000000000000000 000000101000000000000000000000 000000010100000000000000000000 000000001010000000000000000000 000000000101000000000000000000 000000000010100000000000000000 000000000001010000000000000000 000000000000101000000000000000 000000000000010100000000000000 000000000000001010000000000000 000000000000000101000000000000 000000000000000010100000000000 000000000000000001010000000000 000000000000000000101000000000 000000000000000000010100000000 000000000000000000001010000000 000000000000000000000101000000 000000000000000000000010100000 000000000000000000000001010000 000000000000000000000000101000 000000000000000000000000010100 000000000000000000000000001010 000000000000000000000000000101 000000000000000000000000000010