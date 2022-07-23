
import numpy as np
import random
import time


nbr_iter = 100
max_stagnation = 5
max_stagnation_ratio = 0.9
max_init_heur = 1
population_size = 100
mutation_nbre_elements = 2
mutation_ratio = 0.4 #or proba
crossover_ratio = 0.5 #or proba
######################## INIT ########################

def   max_vertex_degree( G ): 
       # trier les sommets de G selon l'ordre décroissant de leurs degrés 
       graph   =    G.adjMatrix 
       degrees   =  [[ j , sum ( i )]  for   i , j   in   zip ( graph  ,  range ( len ( G )))] 
       degrees . sort ( key = lambda   degrees :  degrees [ 1 ]) 
       degrees . reverse ()
       return   degrees[0][1] #degré maximal

def voisins(G, i):
    M = G.adjMatrix
    listeVoisins = []
    n = len(M)
    for j in range(n):
        if M[i][j] == 1:
            listeVoisins.append(j)

    return (listeVoisins)

def generate_random_coloring(G):
    M = G.adjMatrix
    n = len(M)
    max_degree = max_vertex_degree(G)
    coloring = [-1 for j in range(n)]
    for i in range(n):
        unsafe = True
        neighbors = voisins(G, i)
        while unsafe:
            color = random.randint(0,max_degree)
            
            forbiden = [coloring[v] for v in neighbors]
            if not (color in forbiden):
                unsafe = False
                coloring[i] = color

    return coloring


def generate_population(G, nbre, useHeuristic):
    H_max = 0
    population = []
    for i in range(nbre):
        H_random = random.randint(0,nbre*10)
        if useHeuristic and H_max < max_init_heur and H_random < nbre*2:
            # use heuristic
            print("Note: a heuristic solution has been added to the population")
            H_max +=1
            #s = bnb.dfs(G)
        else:
            s = generate_random_coloring(G)
        population.append(s)
    return population

######################## Mutation ########################

def min_safe_color(G,i,S):
    neighbors = voisins(G, i)
    forbiden = [S[v] for v in neighbors]
    forbiden.append(S[i])
    forbiden = np.unique(forbiden)
    max_degree = max_vertex_degree(G)
    safe = [item for item in range(max_degree) if item not in forbiden]
    if len(safe) == 0:
        return S[i]
    else:
        return safe[random.randint(0, len(safe)-1)]
    

def mutation_operator(G,S):
    n = len(S)
    M = S.copy() 
    index = [random.randint(0,n - 1) for i in range(mutation_nbre_elements)]
    unique_index = np.unique(index)
    for j in unique_index:
        M[j] = min_safe_color(G,j,S)
    return M

def mutation(G,population):
    l = len(population)
    nbr_mutation = int(l * mutation_ratio) + 1
    mutations = []
    index = [random.randint(0,l-1) for i in range(nbr_mutation)]
    unique_index = np.unique(index)
    for j in unique_index:
        M = mutation_operator(G,population[j])
        mutations.append(M)
    return mutations


######################## Crossover ########################
def crossover_onepoint(S1,S2):
    l = len(S1)
    E1 = []
    E2 = []
    if l==1:
        return S1,S2
    elif l != 0:
        i = random.randint(1,l-1)
        E1 = S1[0:i]
        E1.extend(S2[i:l])

        E2 = S2[0:i]
        E2.extend(S1[i:l]) 
    return E1,E2

def crossover_twopoints(S1,S2):
    l = len(S1)
    i1 = random.randint(1,l-2)
    i2 = random.randint(i1,l-1)
    print(i1,i2)
    E1 = []
    E2 = []
    if l==1:
        return S1,S2
    elif l==2:
        return S1[0:1].extend(S2[1:2]), S2[0:1].extend(S1[1:2])
    elif l != 0:
        E1 = S1[0:i1]
        E1.extend(S2[i1:i2])
        E1.extend(S1[i2:l])

        E2 = S2[0:i1]
        E2.extend(S1[i1:i2])
        E2.extend(S2[i2:l])

    return E1,E2


def swap(G,S1,S2,i):
    B1 = []
    R1 = []
    B2 = []
    R2= []

    return B1.extend(R1),B2.extend(R2)

def crossover_clean(G,S1,S2):
    l = len(S1)
    E1 = []
    E2 = []
    if l==1:
        return S1,S2
    elif l != 0:
        i = random.randint(1,l-1)
        E1,E2 = swap(G,S1[0:i],S2[0:i],i)
        P1,P2 = swap (G,S2[i:l],S1[i:l],i)
        E1.extend(P1)
        E2.extend(P2) 
    return E1,E2

def crossover(population):
    l = len(population)
    nbr_crossover = int(l * crossover_ratio) + 1
    children = []

    indexP1 = [random.randint(0,l-1) for i in range(nbr_crossover)]
    indexP2 = [random.randint(0,l-1) for i in range(nbr_crossover)]

    couples = [(indexP1[i],indexP2[i]) for i in range(nbr_crossover) if not (indexP1[i] == indexP2[i])]
    
    for couple in couples:
        child = crossover_onepoint(population[couple[0]],population[couple[1]])
        children.extend(child)
    return children

######################## Selection ########################
def check_validity(G,S):
    M = G.adjMatrix
    n = len(M)
    for i in range(n - 1):
        for j in range(n - 1):
            if M[i][j]==1 and j!=i and S[i]==S[j]:
                return False   
    return True

def fitness_validity(G,S):
    valid = check_validity(G,S)
    if not valid:
        r = 0.25
    else:
        r = 1
    l = len(S)
    n = len(np.unique(np.asarray(S)))
    return (l - n) * r

def fitness(S):
    l = len(S)
    n = len(np.unique(np.asarray(S)))
    return l - n

def tournament_selection(population):
    new_population = []
    l = len(population)
    for j in range(2):
        random.shuffle(population)
        for i in range(0, l-1,2):
            if fitness(population[i]) > fitness(population[i+1]):
                new_population.append(population[i])
            elif fitness(population[i]) > fitness(population[i+1]):
                new_population.append(population[i+1])
            else:
                rand = random.randint(0, 1)
                new_population.append(population[i+rand])
    return new_population

######################## Update ########################
def roulette_wheel_selection(G,population,n):
    fitness_list = [fitness_validity(G,S) for S in population]
    Fitness_sum = np.sum(fitness_list)
    probability_list = fitness_list/Fitness_sum
    index = [i for i in range(len(population))]
    index_selection = np.random.choice(index,replace=False, p=probability_list, size=n)
    selection = [population[i] for i in index_selection]
    return selection

def update(G,new_population):
    population = roulette_wheel_selection(G,new_population,population_size)
    return population

######################## Stop ########################
def checkForStagnation(new,old):
    same = [value for value in new if value in old]
    if (len(same)/len(old)) > max_stagnation_ratio:
        return 1
    else:
        return 0




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



def genetic(G):
    count = 0
    stagnation = 0
    M = G.adjMatrix
    n = len(M)
    population = generate_population(G, population_size, False)
    updated= []
    while count < nbr_iter and stagnation < max_stagnation :
        new_population = tournament_selection(population)
        children = crossover(new_population)
        new_population.extend(children)
        mutations = mutation(G,population)
        new_population.extend(mutations)
        updated = update(G,new_population)
        #stagnation += checkForStagnation(updated,population)
        population = updated
    
    print(count,stagnation)
    return population



def main():

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

    
    s = time.time()
    population = genetic(g)
    e = time.time()
    print("Time :", e-s, " seconds")
    population_fitness = [fitness(S) for S in population]
    print("Fitness of resulted population", population_fitness)
    mi = np.argmin(population_fitness)
    print("Best solution found with fitness=", population_fitness[mi], "and ", nb_sommets - population_fitness[mi],  " colors is: ")
    print(population[mi])



main()
