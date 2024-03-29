import numpy as np
import time

class Graph(object):

    # Initialize the matrix
    def __init__(self, size):
        self.adjMatrix = []
        for i in range(size):
            self.adjMatrix.append([0 for i in range(size)])
        self.size = size

    # Add edges
    def add_edge(self, v1, v2):
        if v1 == v2:
            print("Same vertex %d and %d" % (v1, v2))
            return
        self.adjMatrix[v1][v2] = 1
        self.adjMatrix[v2][v1] = 1

    # Remove edges
    def remove_edge(self, v1, v2):
        if self.adjMatrix[v1][v2] == 0:
            print("No edge between %d and %d" % (v1, v2))
            return
        self.adjMatrix[v1][v2] = 0
        self.adjMatrix[v2][v1] = 0

    def __len__(self):
        return self.size

    # Print the matrix
    def print_matrix(self):
        for row in self.adjMatrix:
            for val in row:
                print('')
            print(self.adjMatrix)

# Constructive algorithm (amélioration de greedy algorithm)
# au lieu de séléctionner les vertex aléatoirement
# on choisit le plus grand degre de saturation
def voisins(G, i):
    M = G.adjMatrix
    listeVoisins = []
    n = len(M)
    for j in range(n):
        if M[i][j] == 1:
            listeVoisins.append(j)

    return (listeVoisins)

# degree of saturation 
# checking adjacent nodes and counting the number of colored ones
# Then choosing the node with highest degree
# initially / or when sat_degree is equal we choose the vertex with highest degree
#to minimize interactions we actually need to update saturation
#degree to the neighbors of the newly colored vertex                

def GCP(G):
    #Saturation degrees
    # initially no vetex is colored so all nodes have 0 sat degree
    sat = [0]*len(G.adjMatrix)
    
    #Colors
    colors = [-1]*len(G.adjMatrix)
    nbcolors = 0
    
    #Vertex degrees
    d = [sum(i) for i,j in zip(G.adjMatrix , range(len(G)))]    
    #chosen_nodes = []
    
    while (-1 in colors):
        #Update nb distinct colors used  
        nbcolors =  len(set([i for i in colors if i !=-1]))
        
        #Choose node
        # choose maximum degree 
        # node must have not been colored      
        if sat.count(max(sat))==1 :
           chosen_node = sat.index(max(sat))
        elif max(sat) != 0:
           #choose from the max(sat) set the one with the highest degree
           max_sat = [i for i in range(len(sat)) if sat[i] == max(sat)]
           d_max_sat = [[d[i],i] for i in max_sat]
           chosen_node = [ j for i,j in d_max_sat if i == max([d[i] for i in max_sat]) ][0]
        else :
           chosen_node = d.index(max(d))

        # initilize possible_colors for this node
        possible_colors = [i for i in range(nbcolors)]
    
        #Assign color
        # cnd : use avail colors else new color
        # loop through colored neighbors and check if they've got previous colors
        colored_nodes = [i for i in range(len(colors)) if colors[i] !=-1]
        for colored_node in colored_nodes:
           if colored_node in voisins(G, chosen_node):
              if colors[colored_node] in possible_colors:
                 possible_colors.remove(colors[colored_node])
        # assigning a color to chosen_node
        if possible_colors != [] :
           colors[chosen_node] = possible_colors[0]
        else:
           colors[chosen_node] = nbcolors
        
        #Test 
        #chosen_nodes.append((chosen_node,d[chosen_node],sat[chosen_node]))
        #print("chosen",chosen_nodes)
        
        # update d each iteration by removing colored nodes 
        d[chosen_node] = -1
        sat[chosen_node] = -1
        
        # update saturation
        for voisin in voisins(G, chosen_node):
            if sat[voisin] != -1 :
               sat[voisin] += 1
    
    print("Colors :" , colors)
    print("Num colors :" , nbcolors)


def main():
    # p edge 496 11654 data1 solution optimale : 65, fpsol2.i.1.col
    # p edge 191 2360 data2 solution optimale : 8, myciel7.col
    # p edge 864 18707 data3 solution optimale : 54, inithx.i.1.col
    # p edge 74 602 data4 solution optimale : 11, huck.col
    # p edge 80 508 data5 solution optimale : 10, jean.col
    # p edge 120 1276 data6 solution optimale : 9, games120.col
    # p edge 87 406(812) data6 solution optimale : 11, david.col
    # p edge 23 71 data7 solution optimale : 5, myciel4.col
    # p edge 49, 476(952) data8 solution optimale : 7, queen7_7.col
    M = np.genfromtxt("../Benchmark/data7.txt")
    M = M[:, 1:]
    y = []
    for i in M: 
        y.append(int(i[0]))
        y.append(int(i[1]))

    nb_sommets = max(y)

    g = Graph(nb_sommets)
    for i in M:
        g.add_edge(int(i[0]) - 1, int(i[1]) - 1)

    start_time = time.time()
    GCP(g)
    end_time = time.time()
    ExecTime = end_time - start_time
    print("Temps d'exécution en secondes = ", ExecTime)


if __name__ == '__main__':
    main()

