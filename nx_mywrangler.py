import os
import shutil
import numpy as np
import pandas as pd
import time
import networkx as nx
from sklearn import linear_model
import argparse
import ROOT as r

from tools.selection import GetParticles
from tools.filterGraph import filterGraph
from tools.plottrackreco import plotTrackRecoEff

def getAllPaths(G, starting_nodes, ending_nodes):

    listpath_node_hit_id = []
    listpath_node_r = []
    listpath_node_z = []
    listpath_node_phi = []
    listpath_node_weight = []

    listpath_edge_prediction = []
    listpath_edge_solution = []
    listpath_edge_pt = []

    #list_edges = [ G.edges[(u,v)]['solution'] for u,v,e in G.edges(data=True) ]

    for i in starting_nodes:
        for j in ending_nodes:
            #if((i,j) not in G.edges()):
            #    continue
            for path in nx.all_simple_paths(G, source=i, target=j):
                #print(path)
                #print(nx.path_weight(G, path, "prediction"))
                list_hitid_path = [G.nodes[i]["hit_id"] for i in path]
                list_r_path     = [G.nodes[i]["r"]      for i in path]
                list_z_path     = [G.nodes[i]["z"]      for i in path]
                list_phi_path   = [G.nodes[i]["phi"]    for i in path]
                list_path_weight = nx.path_weight(G, path, "prediction")

                listpath_node_hit_id.append(list_hitid_path)
                listpath_node_r.append(list_r_path)
                listpath_node_z.append(list_z_path)
                listpath_node_phi.append(list_phi_path)
                listpath_node_weight.append(list_path_weight)
                #print(list_hitid_path)

            for path in nx.all_simple_edge_paths(G, source=i, target=j):
                #print(path)
                list_prediction_path = [G.edges[i]["prediction"] for i in path]
                list_solution_path   = [G.edges[i]["solution"]   for i in path]
                list_pt_path         = [G.edges[i]["pt"]         for i in path]

                listpath_edge_prediction.append(list_prediction_path)
                listpath_edge_solution.append(list_solution_path)
                listpath_edge_pt.append(list_pt_path)
                #print(list_prediction_path)

    #return listofpath #, listofsolutionpath
    return listpath_node_hit_id, listpath_node_r, listpath_node_z, listpath_node_phi, listpath_node_weight, listpath_edge_prediction, listpath_edge_solution, listpath_edge_pt


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--Walk', action='store_true', help='run walkthrough algo', default=False)
    parser.add_argument('--LabelComp', action='store_true', help='run label component', default=False)

    args, _ = parser.parse_known_args()

    doWalk  = args.Walk
    doLabComp = args.LabelComp

    frame_particles_with_hits = []

    frame_isolates = []
    frame_start = []
    frame_end = []
    frame_path = []
    frame_path_node = []
    frame_path_edge = []

    frame_one_neighbor = []
    frame_two_neighbors = []
    list_hits_matched = []
    list_hits_matched_particle_iterator = []
    debug = True

    #r.gStyle.SetOptStat(0)
    htruth_pt_perfect = r.TH1D("","",20, 0, 10)
    hreco_pt_perfect = r.TH1D("","",20, 0, 10)

    htruth_eta_perfect = r.TH1D("","",25,-5,5)
    hreco_eta_perfect = r.TH1D("","",25,-5,5)

    htruth_phi_perfect = r.TH1D("","",30,-1,2)
    hreco_phi_perfect = r.TH1D("","",30,-1,2)

    htrack_eff_pt_perfect  = r.TH1D("","",20, 0, 10)
    htrack_eff_eta_perfect  = r.TH1D("","",25,-5,5)
    htrack_eff_phi_perfect  = r.TH1D("","",30,-1,2)

    ## Loop over all events
    for i in range(980,1000):
        print("Processing event #{}".format(i))

        ################################################################################################################
        ################################################################################################################
        ## open truth container
        truth = pd.read_csv("input/event980to1k/event000000{}-truth.csv".format(i))
        ## open particles container
        particles = pd.read_csv("input/event980to1k/event000000{}-particles.csv".format(i))


        ## Get list of particle_ids for particles of interest
        ## GetParticles(hits, particles, ptCut, nHits, etaSlice, phiSlice, etaMin=0, etaMax=2, phiMin=0, phiMax=1):
        list_selected_particle_id = GetParticles(truth, particles, 1000, 3, False, True)

        #print(list_selected_particle_id[0])
        list_selected_hits = [list(truth[truth["particle_id"] == i].hit_id.values) for i in list_selected_particle_id]
        #print(list_selected_hits)
        ## Store particles of interest in dataframe
        all_particles_with_hits = pd.DataFrame()
        all_particles_with_hits["selected_particle_id"] = list_selected_particle_id
        all_particles_with_hits["selected_hit_id"] = list_selected_hits
        #all_particles_with_hits["particles_with_hits_iterator"] = list_hits_matched_particle_iterator
        frame_particles_with_hits.append(all_particles_with_hits)
        #print(list_hits_matched)
        ################################################################################################################
        ################################################################################################################

        ## Let's read the graph now
        ## Load graph from inference (gpickle format)
        G = nx.read_gpickle("data/event{}_reduced_0.0.gpickle".format(i))
        #G = nx.read_gpickle("data/RW2_FW0p2_LR0p0005/event{}_reduced_0.5.gpickle".format(i))


        ############################
        ## Filter graph
        ############################
        ## filterGraph(Graph, prediction_or_solution='solution', threshold=0)
        ## either choose solution with 0
        ## or prediction and a certain threshold
        G = filterGraph(G, 'solution', 0)

        ##################################
        ## Get starting and ending nodes
        ##################################
        ## Get all starting nodes
        starting_nodes = [node for node in G.nodes if G.in_degree(node) == 0]
        starting_nodes_hit_id = [G.nodes[node]['hit_id'] for node in G.nodes if G.in_degree(node) == 0]
        starting_nodes_r = [G.nodes[node]['r'] for node in G.nodes if G.in_degree(node) == 0]
        starting_nodes_z = [G.nodes[node]['z'] for node in G.nodes if G.in_degree(node) == 0]
        starting_nodes_phi = [G.nodes[node]['phi'] for node in G.nodes if G.in_degree(node) == 0]
        all_start = pd.DataFrame()
        all_start["start"] = starting_nodes
        all_start["start_hit_id"] = starting_nodes_hit_id
        all_start["start_r"] = starting_nodes_r
        all_start["start_z"] = starting_nodes_z
        all_start["start_phi"] = starting_nodes_phi
        frame_start.append(all_start)

        ## Get all ending nodes
        ending_nodes = [node for node in G.nodes if G.out_degree(node) == 0]
        ending_nodes_hit_id = [G.nodes[node]['hit_id'] for node in G.nodes if G.out_degree(node) == 0]
        ending_nodes_r = [G.nodes[node]['r'] for node in G.nodes if G.out_degree(node) == 0]
        ending_nodes_z = [G.nodes[node]['z'] for node in G.nodes if G.out_degree(node) == 0]
        ending_nodes_phi = [G.nodes[node]['phi'] for node in G.nodes if G.out_degree(node) == 0]
        all_end = pd.DataFrame()
        all_end["end"] = ending_nodes
        all_end["end_hit_id"] = ending_nodes_hit_id
        all_end["end_r"] = ending_nodes_r
        all_end["end_z"] = ending_nodes_z
        all_end["end_phi"] = ending_nodes_phi
        frame_end.append(all_end)
        print("========================================================================")


        all_path_node = pd.DataFrame()
        all_path_edge = pd.DataFrame()

        if doWalk:
            trackid_node = []
            trackid_edge = []

            list_node_hit_id = []
            list_node_r = []
            list_node_z = []
            list_node_phi = []
            list_node_weight = []
            list_edge_prediction = []
            list_edge_prediction_sum = []
            list_edge_solution = []
            list_edge_pt = []

            listpath_node_hit_id = []
            listpath_node_r = []
            listpath_node_z = []
            listpath_node_phi = []
            listpath_edge_prediction = []
            listpath_edge_solution = []
            listpath_edge_pt = []


            listpath_node_hit_id, listpath_node_r, listpath_node_z, listpath_node_phi, listpath_node_weight, listpath_edge_prediction, listpath_edge_solution, listpath_edge_pt = getAllPaths(G, starting_nodes, ending_nodes)
            #list_of_path = getAllPaths(G, starting_nodes, ending_nodes)

            ## Check the min and max lengths
            if debug:
                max = 0
                min = 1000
                for i in listpath_node_hit_id:
                    if(len(i) > max):
                        max = len(i)
                        if(len(i) < min):
                            min = len(i)
                print("Max = {}, Min = {}".format(max, min))

            for i in range(len(listpath_node_hit_id)):
                for j in range(len(listpath_node_hit_id[i])):
                    trackid_node.append(i)
                    list_node_hit_id.append(listpath_node_hit_id[i][j])
                    list_node_r.append(listpath_node_r[i][j])
                    list_node_z.append(listpath_node_z[i][j])
                    list_node_phi.append(listpath_node_phi[i][j])
                    list_node_weight.append(listpath_node_weight[i])

            for i in range(len(listpath_edge_prediction)):
                sum = np.sum(listpath_edge_prediction[i])
                for j in range(len(listpath_edge_prediction[i])):
                    trackid_edge.append(i)
                    list_edge_prediction.append(listpath_edge_prediction[i][j])
                    list_edge_prediction_sum.append(sum)
                    list_edge_solution.append(listpath_edge_solution[i][j])
                    list_edge_pt.append(listpath_edge_pt[i][j])

                    #print(list_of_path[i][j])
                    #listsolutionpath.append(list_solution_path[i][j])


            all_path_node["path_node_hit_id"] = list_node_hit_id
            all_path_node["track_id_node"] = trackid_node
            #all_path_node["path_node_r"] = list_node_r
            #all_path_node["path_node_z"] = list_node_z
            #all_path_node["path_node_phi"] = list_node_phi
            #all_path_node["path_node_weight"] = list_node_weight

            all_path_edge["track_id_edge"] = trackid_edge
            all_path_edge["path_edge_prediction"] = list_edge_prediction
            all_path_edge["path_edge_prediction_sum"] = list_edge_prediction_sum
            all_path_edge["path_edge_solution"] = list_edge_solution
            all_path_edge["path_edge_pt"] = list_edge_pt
            #all_path["path"] = listpath
            #print(list_node_hit_id)
        if doLabComp:
            print("under construction")

            for c in sorted(nx.connected_components(G)):
                print (c)
            list_of_path = [     len(c)     for c in sorted(nx.weakly_connected_components(G), key=len) ]
            print(list_of_path)
            all_path["path"] = list_of_path

        frame_path_node.append(all_path_node)
        frame_path_edge.append(all_path_edge)



        ################################################################################################################
        ################################################################################################################


        hreco_pt_perfect, htruth_pt_perfect = plotTrackRecoEff(hreco_pt_perfect, htruth_pt_perfect, particles, truth, list_selected_hits, listpath_node_hit_id, 'perfect', 'pt')
        hreco_eta_perfect, htruth_eta_perfect = plotTrackRecoEff(hreco_eta_perfect, htruth_eta_perfect, particles, truth, list_selected_hits, listpath_node_hit_id, 'perfect', 'eta')
        hreco_phi_perfect, htruth_phi_perfect = plotTrackRecoEff(hreco_phi_perfect, htruth_phi_perfect, particles, truth, list_selected_hits, listpath_node_hit_id, 'perfect', 'phi')


        print("\n")

    htrack_eff_pt_perfect.Divide(hreco_pt_perfect, htruth_pt_perfect, 1, 1, "B")
    htrack_eff_eta_perfect.Divide(hreco_eta_perfect, htruth_eta_perfect, 1, 1, "B")
    htrack_eff_phi_perfect.Divide(hreco_phi_perfect, htruth_phi_perfect, 1, 1, "B")

    mean_from_pt  = hreco_pt_perfect.GetEntries() /  htruth_pt_perfect.GetEntries()
    mean_from_eta  = hreco_eta_perfect.GetEntries() /  htruth_eta_perfect.GetEntries()
    mean_from_phi  = hreco_phi_perfect.GetEntries() /  htruth_phi_perfect.GetEntries()

    print("GetMean from pT = {}, from eta = {}, from phi = {}".format(mean_from_pt, mean_from_eta, mean_from_phi))
    f = r.TFile.Open('trackEff.root', 'RECREATE')
    htruth_pt_perfect.Write("truth_pt")
    hreco_pt_perfect.Write("reco_pt")
    htruth_eta_perfect.Write("truth_eta")
    hreco_eta_perfect.Write("reco_eta")
    htruth_phi_perfect.Write("truth_phi")
    hreco_phi_perfect.Write("reco_phi")
    htrack_eff_pt_perfect.Write("trackeff_pt")
    htrack_eff_eta_perfect.Write("trackeff_eta")
    htrack_eff_phi_perfect.Write("trackeff_phi")



    ## Convert list of results to dataframe
    result_particles_with_hits = pd.concat(frame_particles_with_hits)
    result_start = pd.concat(frame_start)
    result_end = pd.concat(frame_end)
    ## Dataframe with output same as first wrangler (2 columns, hit_id, track_id)
    result_path_node = pd.concat(frame_path_node)
    result_path_edge = pd.concat(frame_path_edge)

    ## Save dataframe as csv files
    result_particles_with_hits.to_csv("./nx_particles_with_hits_mywrangler.csv")
    result_start.to_csv("./nx_start_mywrangler.csv")
    result_end.to_csv("./nx_end_mywrangler.csv")
    result_path_node.to_csv("./nx_path_node_mywrangler.csv", index=False)
    result_path_edge.to_csv("./nx_path_edge_mywrangler.csv")

    print("results saved.")



if __name__ == "__main__":
    main()
