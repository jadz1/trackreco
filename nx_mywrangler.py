import os
import shutil
import graph_tool.all as gt
import numpy as np
import pandas as pd
import time
import networkx as nx
from sklearn import linear_model
import argparse

def getPhi(x, y):
    return np.arctan2(y, x)

def GetParticles(hits, particles, ptCut, nHits, phiSlice, phiMin=0, phiMax=1):
    InterestingParticles = particles[(particles.pt >= ptCut) & (particles.barcode<200000)].copy()

    # we need to cut as weel on the number of hits, to do so we can count the number of hits of each particles
    # and then only count one hits on each modules (doesn't matter which one here)
    nhits =  hits.drop_duplicates(subset=['particle_id',
                                      'hardware',
                                      'barrel_endcap',
                                      'layer_disk',
                                      'eta_module',
                                      'phi_module'], keep='last')
    nhits = nhits.groupby("particle_id")["hit_id"].count()
    nhits = nhits.reset_index().rename(columns={"index":"particle_id", "hit_id": "nhit_diffModule"})

    nhits = nhits[nhits.nhit_diffModule>=nHits]

    if phiSlice:
        phihits = getPhi(hits.x, hits.y)
        hits["phi_hits"] = phihits
        hits = hits[(hits.phi_hits>phiMin) & (hits.phi_hits<phiMax)]

    #applied the cuts on the number of hits, the pt and remove secondaries
    hitsInfo = hits.merge(nhits, on="particle_id")
    hitsInfo = hitsInfo.merge(InterestingParticles, on = "particle_id")


    #only keep the hit_id and the particle_id
    list_PIDs = list(pd.unique(hitsInfo.particle_id))

    print("\n============= READ DATA =============")
    print("There are {} particles to reconstruct.".format(len(list_PIDs)))

    return list_PIDs


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
    debug = False
    ## Loop over all events
    for i in range(980,981):
    #for i in range(0,1):
        print("Processing event #{}".format(i))

        ################################################################################################################
        ################################################################################################################
        ## open truth container
        truth = pd.read_csv("input/event980to1k/event000000{}-truth.csv".format(i))
        ## open particles container
        particles = pd.read_csv("input/event980to1k/event000000{}-particles.csv".format(i))


        ## Get list of particle_ids for particles of interest
        list_hits_matched = GetParticles(truth, particles, 1000, 3, True)
        ## Store particles of interest in dataframe
        all_particles_with_hits = pd.DataFrame()
        all_particles_with_hits["particles_with_hits_hit_id"] = list_hits_matched
        #all_particles_with_hits["particles_with_hits_iterator"] = list_hits_matched_particle_iterator
        frame_particles_with_hits.append(all_particles_with_hits)

        ################################################################################################################
        ################################################################################################################

        ## Let's read the graph now
        ## Load graph from inference (gpickle format)
        G = nx.read_gpickle("data/event{}_reduced_0.0.gpickle".format(i))
        #G = nx.read_gpickle("data/RW2_FW0p2_LR0p0005/event{}_reduced_0.5.gpickle".format(i))

        ############################
        ## Filter graph
        ############################
        ## Remove edges above threshold
        list_fake_edges = [(u,v) for u,v,e in G.edges(data=True) if e['solution'] == 0]
        #print(list_fake_edges)
        G.remove_edges_from(list_fake_edges)

        ## Remove isolated nodes
        print("Number of isolated hits = {}".format(nx.number_of_isolates(G)))
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

        if debug:
            ## Put node with one neighbor in a list (we should find all of these later in starting_nodes or ending_nodes)
            Nodes_One_Neighbor = [list(nx.all_neighbors(G, node))[0] for node in G.nodes if (len(list(nx.all_neighbors(G, node))) == 1)]
            Nodes_One_Neighbor_hit_id = [G.nodes[list(nx.all_neighbors(G, node))[0]]['hit_id'] for node in G.nodes if (len(list(nx.all_neighbors(G, node))) == 1)]
            Nodes_One_Neighbor_r = [G.nodes[list(nx.all_neighbors(G, node))[0]]['r'] for node in G.nodes if (len(list(nx.all_neighbors(G, node))) == 1)]
            Nodes_One_Neighbor_z = [G.nodes[list(nx.all_neighbors(G, node))[0]]['z'] for node in G.nodes if (len(list(nx.all_neighbors(G, node))) == 1)]
            Nodes_One_Neighbor_phi = [G.nodes[list(nx.all_neighbors(G, node))[0]]['phi'] for node in G.nodes if (len(list(nx.all_neighbors(G, node))) == 1)]
            all_one_neighbor = pd.DataFrame()
            all_one_neighbor["Nodes_One_Neighbor"] = Nodes_One_Neighbor
            all_one_neighbor["Nodes_One_Neighbor_hit_id"] = Nodes_One_Neighbor_hit_id
            all_one_neighbor["Nodes_One_Neighbor_r"] = Nodes_One_Neighbor_r
            all_one_neighbor["Nodes_One_Neighbor_z"] = Nodes_One_Neighbor_z
            all_one_neighbor["Nodes_One_Neighbor_phi"] = Nodes_One_Neighbor_phi
            frame_one_neighbor.append(all_one_neighbor)

            ## Put node with two neighbors in a list (these should constitue the middle of the track)
            Nodes_Two_Neighbors = [list(nx.all_neighbors(G, node)) for node in G.nodes if (len(list(nx.all_neighbors(G, node))) == 2)]
            Nodes_Two_Neighbors_hit_id = [G.nodes[list(nx.all_neighbors(G, node))[0]]['hit_id'] for node in G.nodes if (len(list(nx.all_neighbors(G, node))) == 2)]
            Nodes_Two_Neighbors_r = [G.nodes[list(nx.all_neighbors(G, node))[0]]['r'] for node in G.nodes if (len(list(nx.all_neighbors(G, node))) == 2)]
            Nodes_Two_Neighbors_z = [G.nodes[list(nx.all_neighbors(G, node))[0]]['z'] for node in G.nodes if (len(list(nx.all_neighbors(G, node))) == 2)]
            Nodes_Two_Neighbors_phi = [G.nodes[list(nx.all_neighbors(G, node))[0]]['phi'] for node in G.nodes if (len(list(nx.all_neighbors(G, node))) == 2)]
            all_two_neighbors = pd.DataFrame()
            all_two_neighbors["Nodes_two_Neighbor"] = Nodes_Two_Neighbors
            all_two_neighbors["Nodes_two_Neighbor_hit_id"] = Nodes_Two_Neighbors_hit_id
            all_two_neighbors["Nodes_two_Neighbor_r"] = Nodes_Two_Neighbors_r
            all_two_neighbors["Nodes_two_Neighbor_z"] = Nodes_Two_Neighbors_z
            all_two_neighbors["Nodes_two_Neighbor_phi"] = Nodes_Two_Neighbors_phi
            frame_two_neighbors.append(all_two_neighbors)

            ## Put node with two neighbors in a list (these should not exist,
                                                     #if they do, they must be in the predicted graph
                                                     #and condition on edge score should be considered)
            Nodes_More_Neighbors = [list(nx.all_neighbors(G, node)) for node in G.nodes if (len(list(nx.all_neighbors(G, node))) > 2)]


            ## Check that all starting nodes are in Nodes_One_Neighbor
            intersect_starting_nodes_OneNeighbor = np.in1d(starting_nodes, Nodes_One_Neighbor)
            #print(intersect_starting_nodes_OneNeighbor)
            ## Check that all ending nodes are in Nodes_One_Neighbor
            intersect_ending_nodes_OneNeighbor = np.in1d(ending_nodes, Nodes_One_Neighbor)
            #print(intersect_ending_nodes_OneNeighbor)



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
                for i in list_of_path:
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

            all_path_node["track_id_node"] = trackid_node
            all_path_node["path_node_hit_id"] = list_node_hit_id
            all_path_node["path_node_r"] = list_node_r
            all_path_node["path_node_z"] = list_node_z
            all_path_node["path_node_phi"] = list_node_phi
            all_path_node["path_node_weight"] = list_node_weight

            all_path_edge["track_id_edge"] = trackid_edge
            all_path_edge["path_edge_prediction"] = list_edge_prediction
            all_path_edge["path_edge_prediction_sum"] = list_edge_prediction_sum
            all_path_edge["path_edge_solution"] = list_edge_solution
            all_path_edge["path_edge_pt"] = list_edge_pt
            #all_path["path"] = listpath

        if doLabComp:
            print("under construction")

            for c in sorted(nx.connected_components(G)):
                print (c)
            list_of_path = [     len(c)     for c in sorted(nx.weakly_connected_components(G), key=len) ]
            print(list_of_path)
            all_path["path"] = list_of_path

        frame_path_node.append(all_path_node)
        frame_path_edge.append(all_path_edge)


    ## Convert list of results to dataframe
    result_particles_with_hits = pd.concat(frame_particles_with_hits)
    result_isolates = pd.concat(frame_isolates)
    result_start = pd.concat(frame_start)
    result_end = pd.concat(frame_end)
    #result_path = pd.concat(frame_path)
    result_path_node = pd.concat(frame_path_node)
    result_path_edge = pd.concat(frame_path_edge)

    if debug:
        result_one_neighbor = pd.concat(frame_one_neighbor)
        result_two_neighbors = pd.concat(frame_two_neighbors)

    ## Save dataframe as csv files   
    result_particles_with_hits.to_csv("./nx_particles_with_hits_mywrangler.csv")
    result_isolates.to_csv("./nx_isolates_mywrangler.csv")
    result_start.to_csv("./nx_start_mywrangler.csv")
    result_end.to_csv("./nx_end_mywrangler.csv")
    #result_path.to_csv("./nx_path_mywrangler.csv")
    result_path_node.to_csv("./nx_path_node_mywrangler.csv")
    result_path_edge.to_csv("./nx_path_edge_mywrangler.csv")

    if debug:
        result_one_neighbor.to_csv("./nx_result_one_neighbor_mywrangler.csv")
        result_two_neighbors.to_csv("./nx_result_two_neighbors_mywrangler.csv")
    #result_track.to_csv("./tracks_inout.csv")
    print("results saved.")



if __name__ == "__main__":
    main()
