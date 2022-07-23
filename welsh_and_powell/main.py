from cmath import e
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

def voisins(G, i): #fonction qui retourne la liste des voisins du sommet i
    M = G.adjMatrix
    listeVoisins = []
    n = len(M)
    for j in range(n):
        if M[i][j] == 1:
            listeVoisins.append(j)

    return (listeVoisins)

####fonction qui prend un graphe en entrée et retourne une coloration de G avec un algo glouton (Welsh & Powel)#########
def ColorGluton(G):
    M = G.adjMatrix
    n = len(M)
    #trier les sommets de G selon l'ordre décroissant de leurs degrés
    # INITIALISATION DE LA LISTE DES DEGRES
    D = []
    # CONSTRUCTION DE LA LISTE DES DEGRES
    for i in range(n):
        d = 0
        # On balaie chaque ligne de la matrice d'adjacence
        for j in range(n):
            # Si un coefficient de la ligne est non nul, on incremente d
            if M[i][j] != 0:
                d += 1
        D.append([i, d])
    # TRI DE LA LISTE DES DEGRES (DANS L'ORDRE DECROISSANT)
    D.sort(key=lambda degre: degre[1])
    D.reverse()
    # COLORATION
    # Initialisation de l'indice des couleurs
    C = 0
    # Initialisation du nombre de sommets colores
    ColoredVertices = 0

    # Boucle principale : on balaie D tant qu'il reste au moins un sommet a colorer
    while ColoredVertices < len(D):
        for i in range(len(D)):
            # On ne s'interesse qu'aux sommets non encore colores
            if len(D[i]) == 2:
                # Le sommet est potentiellement coloriable dans la couleur courante
                ColPoss = True
                # Pour tous les sommets precedant le sommet courant dans la liste D
                for j in range(i):
                    # Si le sommet d'indice j<i dans D est deja  colore avec la couleur C et adjacent au sommet d'indice i
                    # alors le sommet d'indice i ne peut etre colore avec C et on va passer au suivant dans D sans rien faire
                    if len(D[j]) == 3 and D[j][2] == C and M[D[i][0]][D[j][0]] == 1:
                        ColPoss = False
                        break
                # Si on est dans une situation favorable, on colorie le sommet d'indice i dans D avec la couleur courante.
                if ColPoss:
                    D[i].append(C)
                    ColoredVertices += 1
        
        # La liste D a ete balayee, on passe a la couleur suivante
        C +=1
        
    print("nombre de couleurs utilisées pour colorer le graphe G avec Welsh et Powel est :",C)

def main():

    # p edge 74 602 data4 solution optimale : 11, huck.col
    # p edge 87 406(812) data6 solution optimale : 11, david.col
    # p edge 23 71 data7 solution optimale : 5, myciel4.col
    
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
    ColorGluton(g)
    end_time = time.time()
    ExecTime = end_time - start_time
    print("Temps d'exécution en secondes = ", ExecTime)


if __name__ == '__main__':
    main()
