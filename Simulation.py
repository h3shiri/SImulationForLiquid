
import os
import sys

file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)

import networkx as nx
import itertools as it

from mesa.space import NetworkGrid
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pylab import meshgrid,cm,imshow,contour,clabel,colorbar,axis,title,show

# from .Node import SimpleNode
import sys

"""

probability_index = sys.argv[1]
NumberOfNodes = sys.argv[2]


"""
#Aka the probability we choose to sample for our nodes
MAGIC_Prob_Setting = 0.8

Number_of_satelite_nodes = 12

STEP = 1
ERROR = -1
ZERO = 0

#Ledger to get different nodes competences
COMPETENCE_KEY = 'competence'
VOTES_KEY = 'votes'
LIQUID_VOTES_KEY = 'liquid'

class StarGraph:

    def __init__(self, magic_number_of_nodes, probability_setting = MAGIC_Prob_Setting):
        #The Network Object
        # self.G = nx.erdos_renyi_graph(12, 0.6)
        self.G = nx.star_graph(n=int(magic_number_of_nodes))
        #Modyfing for extra parameters
        for i in range(int(magic_number_of_nodes) + 1):
            self.G.nodes[i][COMPETENCE_KEY] = np.random.uniform(probability_setting, 1)
            # Whats the difference here ?
            self.G.nodes[i][VOTES_KEY] = 1
            self.G.nodes[i][LIQUID_VOTES_KEY] = 1

            # One extra for the centre plus n neighbours
            self.n = magic_number_of_nodes + 1
            #This shall be our approval graph relations
            self.A = nx.DiGraph()

    # Currently This holds a directed graph with all the approval relations
    # No strict alpha margin has been implemented yet
    def buildApprovalGraph(self):
        # Adding approved neighbours
        self.A = nx.DiGraph()
        for i in self.G.nodes:

            # Select multiple edges
            for j in self.G.neighbors(i):
                # In case of no neighbors
                if self.G.nodes[i][COMPETENCE_KEY] < self.G.nodes[j][COMPETENCE_KEY]:
                    self.A.add_edge(i, j, color='blue')

    #The function is used to traverse next hop based on competence filters
    """
    @:param - node_index - the relevant node index
    :return - relevant neighbour with maximum competence if one exists else -1.
    """
    def traverseGraphByMaxCompetence(self, node_index):
        EMPTY = 0
        if len(list(self.A.neighbors(node_index))) == EMPTY:
            return ERROR
        m = np.argmax([self.G.nodes[j][COMPETENCE_KEY] for j in self.A.neighbors(node_index)])
        # Returns the relevant node form the neighbors
        next_hop_node = [s for s in self.A.neighbors(node_index)][m]
        return next_hop_node

    #Basic Testing in Jupyter seems fine

    """
    @param node_index - essentially what we look for when running the function
    """
    def fetchTraverseNode(self, node_index):
        res = ERROR
        Next_node = node_index

        #Test for neighbours existence.
        if (self.traverseGraphByMaxCompetence(node_index) == ERROR):
            return node_index

        while(Next_node != ERROR):
            res = Next_node
            Next_node = self.traverseGraphByMaxCompetence(Next_node)

        return res

    # Routine for updating the tokens
    def updateVotesTokens(self):
        for i in self.G.nodes:
            target = self.fetchTraverseNode(i)
            #If it's the same we need not move the tokens.
            if i != target:
                self.G.nodes[target][VOTES_KEY] += self.G.nodes[i][VOTES_KEY]
                self.G.nodes[i][VOTES_KEY] = ZERO

    # Set fixed competence for micro calculation
    def setCompetence(self, probability):
        for i in range(self.n):
            self.G.nodes[i][COMPETENCE_KEY] = probability


    def getCompetence(self, ID):
        return self.g.nodes[ID][COMPETENCE_KEY]

    #Tricky needs careful implementation

    #Check Majority Probability f voting pattern
    def majorityProb(self):
        DICT = 1
        competence_list = [x[DICT][COMPETENCE_KEY] for x in self.G.nodes(data=True)]
        return Majority_function(competence_list)

    #TODO: Implement this function
    #consider how to solve this architecture issue
    def liquidProb(self):
        DICT = 1
        competence_list = [x[DICT][COMPETENCE_KEY] for x in self.G.nodes(data=True)]
        votes_list = [x[DICT][VOTES_KEY] for x in self.G.nodes(data=True)]
        return Liquid_function(votes_list, competence_list)

    #The standard gain notation form procassia, Liquid - direct.
    def gain(self):
        return self.liquidProb() - self.majorityProb()

### This is one hell of an ugly solution in order to by pass relative imports issues.


# Use to Measure competence levels, and other magic constants
DEFAULT_COMPETENCE = 0.5

# The significance of higher competence
ALPHA = 0.1

ACTIVE = 1

# Useful iterating functions for predicting coalitions outcomes.

import itertools as it

# self.competence = DEFAULT_COMPETENCE

CORRECTION = 0.01

# return all subset
def all_subsets(ss):
    return it.chain(*map(lambda x: it.combinations(ss, x), range(0, len(ss) + 1)))

# Calculating the probability of an outcome occurring based on this coalition
def prob_coalition(sub_index_list, competencelist):
    res = 1
    for index in range(len(competencelist)):
        if index in sub_index_list:
            res *= competencelist[index]
        else:
            res *= (1 - competencelist[index])
    return res

#Liquid_version so no need accounting for the non active players
def prob_liquid_coalition(sub_index_list, active_players_list, competencelist):
    res = 1
    for index in active_players_list:
        if index in sub_index_list:
            res *= competencelist[index]
        else:
            res *= (1 - competencelist[index])
    return res


# Retuning all possible subselection of indices
def Combo(items):
    All_possible_indices = []
    for L in range(0, len(items) + 1):
        for subset in it.combinations(items, L):
            All_possible_indices.append(subset)
    return All_possible_indices

"""
This function calculates the probability of the majority getting hold
@competence_list - is an array of the various competencies of the various players.
"""

# Tested
def Majority_function(competence_list):
    res = 0
    # In case of a single player
    if len(competence_list) == 1:
        return competence_list[0]
    n = len(competence_list)
    # return possible sub_coalition indices
    sub_selection_of_indices = Combo(range(n))
    for col in sub_selection_of_indices:
        # majority_condition
        if len(col) > (n / 2):
            res += prob_coalition(col, competence_list)
    return res

#The function for the liquid variant

#Tested - for small cases fine
def Liquid_function(votes_list, competence_list):
    res = 0
    # In case of a single player
    if len(competence_list) == 1:
        return competence_list[0]
    n = len(competence_list)
    # return possible sub_coalition indices of active voters
    active_voters = []
    for i in range(n):
        if votes_list[i] > ZERO:
            active_voters.append(i)
    # Sub_coalitions_of_non_zero_weight_voters
    sub_selection_of_indices = Combo(active_voters)
    # Calculating probability of each occurring
    for col in sub_selection_of_indices:
        # majority_votes_condition
        sum = 0
        for i in col:
            sum += votes_list[i]
        #TODO: tie breaking conditions
        if sum > (n / 2):
            res += prob_liquid_coalition(col, active_voters, competence_list)
    return res

# Running simulation
"""
Starting with graph of size m to sizes n and how that effect the gain in the graph
Running on odd sizes only
@:return - graph approximated changes.
"""
def testNumberOfNodesAffect(m,n):
    ODD_JUMP = 2
    results = []
    sizes = list(range(m,n,ODD_JUMP))

    for n in range(m,n,ODD_JUMP):

        results.append(testMicro(50, n=m))

    plt.scatter(sizes, results, color='blue')
    plt.xlabel("number of star nodes")
    plt.ylabel("Gain_by_liquid")
    red_patch = mpatches.Patch(color='blue', label='testing star size')
    plt.legend(handles=[red_patch], loc=2)


    plt.show()

# we approximate to a 100 frac
"""
@:param p1 - form 1 to Caibration_value
@:param p2 - form 1 to 100, strictly bigger then p2
@:param resolution
@:return - graph approximations

"""
def testProbabilityAffect(p1, p2, resolution):
    ODD_JUMP = 1
    results = []
    probs = []
    for pr in np.arange(p1,p2, resolution):
        probs.append(pr)
        results.append(testMicro(100, n=12, p=pr))
    plt.scatter(probs, results, color='green')


    plt.xlabel("Probability")
    plt.ylabel("Gain_by_liquid")
    red_patch = mpatches.Patch(color='green', label='testing competence levels')
    plt.legend(handles=[red_patch], loc=2)
    fig = plt.figure()
    fig.patch.set_facecolor('black')
    plt.show()


#Micro_testing_for_a_given_Setting
"""
n - number of sat nodes, default 12 which makes in total 13
p - the probability setting, default 0.8
iterations - number of times to average the results
@:return - the approximated probability
"""
def testMicro(iterations, n = Number_of_satelite_nodes, p = MAGIC_Prob_Setting):
    avg = 0
    for i in range(iterations):
        Star = StarGraph(n, p)
        Star.buildApprovalGraph()
        Star.updateVotesTokens()
        avg += Star.gain()
    res = avg/iterations
    return res

def testMajority(iterations, n = Number_of_satelite_nodes, p = MAGIC_Prob_Setting):
    avg = 0
    for i in range(iterations):
        Star = StarGraph(n, p)
        avg += Star.majorityProb()
    res = avg/iterations
    return res

# Util for feeding the model.
"""
Arguments order here is th oposite and we assume 100 independent samples.
"""
def testFix(probability, numberOfPoints):
    return testMicro(100, n=numberOfPoints, p=probability)

# check if possible to render in without building all this expensive objects
def extensiveModelRendering():
    # Data
    x = np.arange(0.5, 0.6, 0.005)
    y = np.arange(3, 43, 2)
    X, Y = np.meshgrid(x, y)
    # grid of point
    # ax = fig.gca(projection='3d')

    Z = np.array([testFix(x, y) for x, y in zip(X.flatten(), Y.flatten())])  # use of custom function
    Z = Z.reshape(X.shape)

    fig = plt.figure()
    ax = fig.axes(projection='3d')

    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)

    ax.scatter3D(X, Y, Z, c=Z, cmap='Greens');
    # Customize the z axis.
    ax.set_zlim(-1.01, 1.01)

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()
    #
    # ax.contour3D(X, Y, Z, 50, cmap='binary')
    # ax.set_xlabel('proabilities')
    # ax.set_ylabel('number of nodes')
    # ax.set_zlabel('Gain by liquid')
    # ax.view_init(60, 35)
    #
    # plt.show()
    print("done")

def SilyCaculation():
    star = StarGraph(12)
    star.setCompetence(0.9)
    x = star.majorityProb()


def main():

    # print(testMajority(50, 12, 0.8))
    # testNumberOfNodesAffect(3, 301)
    # testProbabilityAffect(0.5, 0.9, 0.005)
    # testProbabilityAffect(0.8, 0.95, 0.001)
    # testProbabilityAffect(0.48,0.52,0.0005)

    extensiveModelRendering()




if __name__ == '__main__':
    main()
