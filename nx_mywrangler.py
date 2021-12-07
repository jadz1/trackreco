import os
import shutil
import graph_tool.all as gt
import numpy as np
import pandas as pd
import time
import networkx as nx
from sklearn import linear_model
import argparse


def getAllPaths(G, starting_nodes, ending_nodes):
    listofpath = []
    #for i in starting_nodes:
    #    paths = nx.all_simple_paths(G, i, ending_nodes)
    #    listofpath.append(paths)
    for i in starting_nodes:
        for j in ending_nodes:
            for path in nx.all_simple_paths(G, source=i, target=j):
                #print(path)
                list_hitid_path = [G.nodes[i]["hit_id"] for i in path]
                print(list_hitid_path)
                listofpath.append(list_hitid_path)
    return listofpath


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--Walk', action='store_true', help='run walkthrough algo', default=False)
    parser.add_argument('--LabelComp', action='store_true', help='run label component', default=False)

    args, _ = parser.parse_known_args()

    doWalk  = args.Walk
    doLabComp = args.LabelComp


    frame_isolates = []
    frame_start = []
    frame_end = []
    frame_path = []

    debug = False
    ## Loop over all events
    #for i in range(980,981):
    for i in range(0,1):
        print("Processing event #{}".format(i))

        ## Load graph from inference (gpickle format)
        #G = nx.read_gpickle("data/event{}_reduced_0.0.gpickle".format(i))
        G = nx.read_gpickle("data/RW2_FW0p2_LR0p0005/event{}_reduced_0.5.gpickle".format(i))


        ############################
        ## Filter graph
        ############################
        ## Remove edges above threshold
        list_fake_edges = [(u,v) for u,v,e in G.edges(data=True) if e['solution'] == 0]
        #print(list_fake_edges)
        G.remove_edges_from(list_fake_edges)

        ## Remove isolated nodes
        #print(nx.number_of_isolates(G))
        #isolates = np.array(nx.isolates(G))
        isolates=[node for node in G.nodes if G.degree(node) == 0]
        #print(isolates)
        G.remove_nodes_from(list(nx.isolates(G)))

        all_isolates = pd.DataFrame()
        all_isolates["isolates"] = isolates
        frame_isolates.append(all_isolates)

        ##################################
        ## Get starting and ending nodes
        ##################################
        ## Get all starting nodes
        starting_nodes = [node for node in G.nodes if G.in_degree(node) == 0]
        all_start = pd.DataFrame()
        all_start["start"] = starting_nodes
        frame_start.append(all_start)

        ## Get all ending nodes
        ending_nodes = [node for node in G.nodes if G.out_degree(node) == 0]
        all_end = pd.DataFrame()
        all_end["end"] = ending_nodes
        frame_end.append(all_end)
        print("========================================================================")



        all_path = pd.DataFrame()

        if doWalk:
            list_of_path = getAllPaths(G, starting_nodes, ending_nodes)
            #print(list_of_path)
            ## Check the min and max lengths
            if debug:
                max = 0
                min = 1000
                for i in list_of_path:
                    if(len(i) > max):
                        max = len(i)
                        if(len(i) < min):
                            min = len(i)
                print("Max = {}, Min = {}".format(max, min))
            all_path["path"] = list_of_path

        if doLabComp:
            print("under construction")

        frame_path.append(all_path)

    ## Convert list of results to dataframe

    result_isolates = pd.concat(frame_isolates)
    result_start = pd.concat(frame_start)
    result_end = pd.concat(frame_end)
    result_path = pd.concat(frame_path)

    result_isolates.to_csv("./nx_isolates_mywrangler.csv")
    result_start.to_csv("./nx_start_mywrangler.csv")
    result_end.to_csv("./nx_end_mywrangler.csv")
    result_path.to_csv("./nx_path_mywrangler.csv")
    #result_track.to_csv("./tracks_inout.csv")
    print("results saved.")



if __name__ == "__main__":
    main()
