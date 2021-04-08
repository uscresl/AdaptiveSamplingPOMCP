# Auxilliary code

import numpy as np 
from itertools import chain, combinations
from statistics import mean
import numba

def average_or_0_on_empty(l):
    if l != []:
        return np.average(l)
    else:
        return 0

# builds a tree
class BuildTree():
    def __init__(self, giveParameters = ['isRoot', {}, 0, 0, []] ,start_states=[]): #add init dist

        # index for nodes
        self.count = -1
        self.nodes = {}
        # dictionary where key is node and value is list of corresponding values
        # = [ parent, children, Nc, Value, B() (-1) if action node]

        self.giveParameters = []

        # Create instance attributes for initialization
        # This is necessary to avoid multiple instances sharing same
        # Attributes
        giveParameters[-1] = start_states
        for i in giveParameters:
            if type(i) == str or type(i) == int:
                self.giveParameters.append(i)
            else:
                self.giveParameters.append(i.copy())
        self.nodes[self.count] = self.giveParameters

    # Expand the tree by one node. 
    # If the result of an action give IsAction = True
    def ExpandTreeFrom(self, parent, index, IsAction = False):
        self.count += 1
        if IsAction: 
            # add node to tree
            # We store all rewards so that we can use it as a stopping metric
            self.nodes[self.count] = [parent, {}, 0, [], -1]
            # inform parent node
            self.nodes[parent][1][index] = self.count 
        else:
            self.nodes[self.count] = [parent, {}, 0, 0, []] 
            self.nodes[parent][1][index] = self.count

    # Check given nodeindex corresponds to leaf node
    def isLeafNode(self, n):
        return self.nodes[n][2] == 0

    def isActionNode(self, n):
        return self.nodes[n][4] == -1

    def get_node_reward(self, n):
        if isinstance(n, int):
            n = self.nodes[n]

        if n[4] == -1:  # Action node
            return mean(n[3])
        else:
            return 0

    def get_reward_history(self, n):
        return self.nodes[n][3]

    # note: leaf nodes won't be unvisited in his current POMCP tree implementation (this should change)
    def isUnvisitedObsNode(self, n):
        return self.nodes[n][4] == []

    def get_children(self, n):
        return self.nodes[n][1]

    def get_value(self, n):
        return average_or_0_on_empty(self.nodes[n][3])

    def get_value_stdv(self, n):
        l = self.nodes[n][3]
        if l != []:
            return np.std(l)
        else:
            return 0

    def get_visits(self, n):
        return self.nodes[n][2]

    def get_parent(self, n):
        return self.nodes[self.nodes[n][0]]

    # As in pomcp/ Checks that an observation was already made before moving
    def getObservationNode(self,h,sample_observation):
        # Check if a given observation node has been visited
        if sample_observation not in list(self.nodes[h][1].keys()):
            # If not create the node
            self.ExpandTreeFrom(h, sample_observation)
        # Get the nodes index
        Next_node = self.nodes[h][1][sample_observation]
        return Next_node

    # Removes a node and 
    def prune(self, node):
        children = self.nodes[node][1]
        del self.nodes[node] 
        for _, child in children.items():
            self.prune(child)

    # make new root and update children
    def make_new_root(self, new_root):
        self.nodes[-1] = self.nodes[new_root].copy()
        del self.nodes[new_root]
        self.nodes[-1][0] = 'isRoot'
        # update children
        for _ , child in self.nodes[-1][1].items():
            self.nodes[child][0] = -1

    # Prune tree after action and observation were made
    def prune_after_action(self, action, observation):
        # Get node after action
        action_node = self.nodes[-1][1][action]

        # Get new root (after observation)
        new_root = self.getObservationNode(action_node, observation)

        # remove new_root from parent's children to avoid deletion
        del self.nodes[action_node][1][observation]

        # prune unnesecary nodes
        self.prune(-1)

        # set new_root as root (key = -1)
        self.make_new_root(new_root)

#UCB score calculation
@numba.jit(nopython=True)
def UCB(N,n,V,c = 1): #N=Total, n= local, V = value, c = parameter
    return V + c*np.sqrt(np.log(N)/n)

# from itertools recipes
# creates power set
def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))
