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

def voisins(G, i):
    M = G.adjMatrix
    listeVoisins = []
    n = len(M)
    for j in range(n):
        if M[i][j] == 1:
            listeVoisins.append(j)

    return (listeVoisins)

def trier(G):
    # trier les sommets de G selon l'ordre décroissant de leurs degrés
    # CONSTRUCTION DE LA LISTE DES DEGRES
    graph =  G.adjMatrix
    degrees = [[j,sum(i)] for i,j in zip(graph , range(len(G)))]
    # TRI DE LA LISTE DES DEGRES (DANS L'ORDRE DECROISSANT)
    degrees.sort(key=lambda degrees: degrees[1])
    degrees.reverse()
    
    D = list(np.array(degrees)[:,0])
    return D

def removeNeighboringColors(G, sommet, colorsPossible, colorsAlreadyUsed):
    v=voisins(G, sommet)
    for j in range(len(v)):
            if (colorsAlreadyUsed[v[j] ]!=-1 and (colorsAlreadyUsed[v[j]] in colorsPossible)):
                colorsPossible.remove(colorsAlreadyUsed[v[j]] ) 

def nextPossibleColor(G, sommet, lastCheckedColor, verticesAlreadyColored, upperbound):
    v = voisins(G, sommet)
    nextColorIsFound = False

    for nextColor in range(lastCheckedColor + 1, upperbound + 1):
        if (not nextColorIsFound) :
            nextColorIsFound = True
            for voisin in v:
                if (verticesAlreadyColored[voisin] == nextColor):
                    nextColorIsFound = False
                    break
            if nextColorIsFound : return nextColor
    return -1

def GCPbranchbound(G):
    #trier les sommets du graphe selon l'ordre décroissant de leurs degrés
    D = trier(G)
    Update = True
    nbresommet = len(G.adjMatrix)
    i = 1 
    #initialiser la borne inf au nbr de sommets
    upperbound = nbresommet
    #var intermédiaire pour mettre à jour les bornes
    L = [-1 for i in range(nbresommet)]
    #on a encore vérifié aucune couleur
    lastCheckedColor = [0 for i in range(nbresommet)]
    colors = [-1 for i in range(nbresommet)]
    L[0] = 1
    #initialiser la borne inf à 1
    lowerbound = 1  # couleur
    #initialiser toutes les couleurs à null
    #assigner la couleur 1 au 1er sommet
    colors[D[0]] = 1

    while i > 0: #génerer des solutions tant que la racine n'est pas encore atteinte
        
        if Update: #calculer l'ensemble U contenant les couleurs possibles (on enlève celles des voisins)
            colorToCheck = nextPossibleColor(G, D[i], lastCheckedColor[D[i]], colors, upperbound)
        
        if colorToCheck == -1: #si pas de couleur possible pour le sommet on remonte et on met à jour la borne inf
            i = i - 1
            lowerbound  = L[D[i]]
            Update = False
        else: #sinon, on affecte au sommet la plus petite couleur possible à partir de l'ensemble U
            if colors[D[i]] == -1 :
                j = nextPossibleColor(G, D[i], lastCheckedColor[D[i]], colors, upperbound)
                colors[D[i]] = j
                lastCheckedColor[D[i]] = j
    
            if (j < upperbound): #tester si la couleur affectée < la borne sup
                if (j > lowerbound ):#tester si la couleur affectée > la borne inf
                    lowerbound = lowerbound  + 1 #mettre à jour la borne inf après l'ajour du couleur du sommet
                if (i == nbresommet - 1): #si on a coloré tous les sommets stocker la solution trouvée et mettre la borne sup = la borne inf
                    upperbound = lowerbound
                    for m in range(nbresommet): #trouver le premier sommet vm tq colors[vm]=borne sup
                        if (colors[D[m]] == upperbound):
                            break
                    i = m - 1 #accéder au sommet précedant vm
                    lowerbound = upperbound - 1 #mettre à jour la borne inf
                    Update = False
                else: # si on a pas encore atteint le nombre de sommets
                    L[D[i]] = lowerbound #stocker la borne inf
                    i = i + 1 #un nouveau sommet est sélectionné pour etre coloré
                    Update = True
            else: #si la couleur affectée > la borne sup
                i = i - 1 #élager la branche actuelle et remonter
                lowerbound = L[D[i]] #remettre la dernière valeur de la borne sup
                Update = False

    diffcolors = [] #calculer le nombre de couleurs utilisées
    for i in (colors):
        if i not in diffcolors:
            diffcolors.append(i)
    print("Num colors : ", len(diffcolors))

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
    GCPbranchbound(g)
    end_time = time.time()
    ExecTime = end_time - start_time
    print("Temps d'exécution en secondes = ", ExecTime)

if __name__ == '__main__':
    main()

