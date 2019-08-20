from skbio import DistanceMatrix
from skbio.tree import nj, TreeNode
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout

class NJTree:

    def __init__(self, dist_matrix):
        self.dist_matrix = dist_matrix
        nr_elements = self.dist_matrix.nr_elements
        self.matrix = []
        for i in range(nr_elements):
            row = []
            for j in range(nr_elements):
                row.append(self.dist_matrix.get_distance(i, j))
            self.matrix.append(row)
        self.ids = list(map(str, self.dist_matrix.labels))
        self.nj_dm = DistanceMatrix(self.matrix, self.ids)
        tree = nj(self.nj_dm)
        self.ids = []
        self.sources = []
        self.targets = []
        self.weights = []
        self.colors = []
        self.node_size = []
        self.virtual_nodes = 0
        self.shown_labels = {}
        self.font_colors = []

        # true #00A693 -- false #CC3333
        for node in tree.preorder():
            name_str = ''
            if node.name is None:
                self.virtual_nodes = self.virtual_nodes + 1
                name_str = 'v'+str(self.virtual_nodes)
                node.name = name_str
                self.ids.append(node.name)
                self.colors.append("black")
                self.node_size.append(20)
                self.shown_labels[str(name_str)] = ""
                self.font_colors.append('k')
            else:
                name = node.name.rsplit(' ', 1)
                if len(name) > 1:
                    node.name =name[1]
                    name2 = name[0].rsplit(' ', 1)
                    if len(name2) > 1:
                        node.name = name2[1]+name[1]
                name = node.name    
                if  name in []:
                    self.ids.append(node.name)
                    self.colors.append("#CC3333")
                    self.node_size.append(800)
                    name_str = node.name
                    self.shown_labels[str(name_str)] = name_str
                else:
                    self.ids.append(node.name)
                    self.colors.append("#00A693")
                    self.node_size.append(800)
                    name_str = node.name
                    self.shown_labels[str(name_str)] = name_str

        for node in tree.preorder():
            for child in node.children:
                self.sources.append(str(node.name))
                self.targets.append(str(child.name))
                self.weights.append(str(child.length))
                
        #G = nx.MultiGraph()
        #for i in range(0, len(self.sources)):
        #    G.add_edge(self.sources[i], self.targets[i])
        #G.nodes()
        # And a data frame with characteristics for your nodes
        #carac = pd.DataFrame(
        #    {'ID': self.ids, 'colors': self.colors, 'node_size': self.node_size})

        #carac = carac.set_index('ID')
        #carac = carac.reindex(G.nodes())
        #plt.figure(1,figsize=(30,20))
        # Plot trees
        # pos=graphviz_layout(G, prog='dot') #dot #
        # self.node_positions = pos
        
        # nx.draw(G, pos, with_labels=True, arrows=False, 
        #        node_size=carac['node_size'], node_color=carac['colors'],
        #        labels = self.shown_labels, font_size=12, font_color="black")
 
        # plt.savefig('draw_trees_with_pygraphviz.png', bbox_inches='tight')   
        # plt.show()
        
        
        
