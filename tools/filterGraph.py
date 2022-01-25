import os
import shutil
import numpy as np
import pandas as pd
import time
import networkx as nx
#from sklearn import linear_model
import argparse
import ROOT as r

def filterGraph(Graph, prediction_or_solution='solution', threshold=1):

    list_fake_edges = [(u,v) for u,v,e in Graph.edges(data=True) if e[prediction_or_solution] < threshold]
    Graph.remove_edges_from(list_fake_edges)

    print("Number of isolated hits = {}".format(nx.number_of_isolates(Graph)))
    Graph.remove_nodes_from(list(nx.isolates(Graph)))

    return Graph
