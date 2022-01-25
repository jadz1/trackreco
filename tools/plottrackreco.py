import os
import shutil
import numpy as np
import pandas as pd
import time
import networkx as nx
#from sklearn import linear_model
import argparse
import ROOT as r

##### IN THIS CODE, THE SPLIT CLUSTER CONDITION SHOULD BE ADDED!!
### If two paths have the same starting and ending nodes, we should choose one and only one

def getPhi(x, y):
    return np.arctan2(y, x)


def plotTrackRecoEff(hreco, htruth, particles, truth, list_selected_hits, listpath_node_hit_id, matchingCriterion='perfect', variable='pt'):

    for itruth in list_selected_hits:
        particle_pid = truth[truth["hit_id"] == itruth[0]].particle_id.values[0]
        ## Make sure we only take particles associated to hits
        if(len(list(particles[particles["particle_id"] == particle_pid].pt)) == 0):
            continue
        ## remove split clusters
        true_PID_hits = truth[truth["particle_id"] == particle_pid]
        #print(true_PID_hits)
        liste_true_hits = true_PID_hits.drop_duplicates(subset=['particle_id','hardware','barrel_endcap','layer_disk','eta_module','phi_module'],keep=False).hit_id.to_list()
        liste_true_hits = sorted(liste_true_hits)
        itruth = sorted(itruth)
        if liste_true_hits != itruth:
            continue
        pt = particles[particles["particle_id"] == particle_pid].pt.values[0]
        eta = particles[particles["particle_id"] == particle_pid].eta.values[0]
        phi = getPhi(particles[particles["particle_id"] == particle_pid].px.values[0] ,  particles[particles["particle_id"] == particle_pid].py.values[0])
        #print(pt)
        if variable == 'pt': htruth.Fill(pt/1000.)
        if variable == 'eta': htruth.Fill(eta)
        if variable == 'phi': htruth.Fill(phi)

        for ireco in listpath_node_hit_id:
            if(len(np.intersect1d(ireco, itruth )) != 0):
                particle_pid = truth[truth["hit_id"] == itruth[0]].particle_id.values[0]
                pt = particles[particles["particle_id"] == particle_pid].pt.values[0]
                eta = particles[particles["particle_id"] == particle_pid].eta.values[0]
                phi = getPhi(particles[particles["particle_id"] == particle_pid].px.values[0] ,  particles[particles["particle_id"] == particle_pid].py.values[0])
                if matchingCriterion=='perfect':
                    #remove = RemoveSplitCluster(list_selected_hits, truth)
                    #if(len(np.intersect1d(ireco, itruth)) == len(itruth) and remove==True):
                    if(len(np.intersect1d(ireco, itruth)) == len(itruth)): ## perfect matching 
                        if variable == 'pt': hreco.Fill(pt/1000.)
                        if variable == 'eta': hreco.Fill(eta)
                        if variable == 'phi': hreco.Fill(phi)
    return hreco, htruth
