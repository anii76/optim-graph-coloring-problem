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


def GCPbranchbound(G):
    #print(G.adjMatrix)
    ColorsPossib = []
    L = []
    colors = []
    #trier les sommets du graphe selon l'ordre décroissant de leurs degrés
    D = trier(G)
    Update = True
    nbresommet = len(G.adjMatrix)
    i = 1
    #initialiser la borne inf au nbr de sommets
    upperbound = nbresommet
    #var intermédiaire pour mettre à jour les bornes
    L = [-1 for i in range(nbresommet)]
    ColorsPossib = [[] for i in range(nbresommet)]
    colors = [-1 for i in range(nbresommet)]
    L[0] = 1
    #initialiser la borne inf à 1
    lowerbound = 1  # couleur
    #initialiser toutes les couleurs à null
    ColorsPossib.append(-1)
    #assigner la couleur 1 au 1er sommet
    colors[D[0]] = 1
    #print(D)

    while i > 0: #génerer des solutions tant que la racine n'est pas encore atteinte
        if Update: #calculer l'ensemble U contenant les couleurs possibles (on enlève celles des voisins)
            ColorsPossib[D[i]] = [j + 1 for j in range(lowerbound  + 1)]
            v = voisins(G, D[i]) #génerer les voisins du sommet
            for j in range(len(v)):
                if colors[v[j]] != -1 and (colors[v[j]] in ColorsPossib[D[i]]):
                    ColorsPossib[D[i]].remove(colors[v[j]])

        if ColorsPossib[D[i]] == []: #si pas de couleur possible pour le sommet on remonte et on met à jour la borne inf
            i = i - 1
            lowerbound  = L[D[i]]
            Update = False
        else: #sinon, on affecte au sommet la plus petite couleur possible à partir de l'ensemble U
            j = ColorsPossib[D[i]][0]
            if colors[D[i]] == -1:
               colors[D[i]] = j
            # print('cc i=',i,'   ',j)
            ColorsPossib[D[i]].remove(j) #retirer la couleur affectée au sommet de l'ensemble U
            # print('k=',k)
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
    print('colors', colors)
    diffcolors = [] #calculer le nombre de couleurs utilisées
    for i in (colors):
        if i not in diffcolors:
            diffcolors.append(i)
    print("le nombre de couleurs utilisées est: ", len(diffcolors))


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
    M = np.genfromtxt("./data7.txt")
    M = M[:, 1:]
    nb_sommets = len(M)
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

