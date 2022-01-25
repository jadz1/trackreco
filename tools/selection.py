import os
import shutil
import numpy as np
import pandas as pd
import time
import networkx as nx
#from sklearn import linear_model
import argparse
import ROOT as r

def getPhi(x, y):
    return np.arctan2(y, x)

def getR(x, y):
    return np.sqrt(x**2 + y**2)

def getEta(x, y, z):
    r = getR(x,y);
    r3 = np.sqrt(r*r + z*z);
    theta = np.arcos(z/r3);

    return -np.log(np.tan(theta*0.5));

def GetParticles(hits, particles, ptCut, nHits, etaSlice, phiSlice, etaMin=0, etaMax=2, phiMin=0, phiMax=1):
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

    if etaSlice:
        etahits = np.absolute(getEta(hits.x, hits.y, hits.z))
        hits["eta_hits"] = etahits
        hits = hits[(hits.eta_hits>etaMin) & (hits.eta_hits<etaMax)]

    if phiSlice:
        phihits = getPhi(hits.x, hits.y)
        hits["phi_hits"] = phihits
        hits = hits[(hits.phi_hits>phiMin) & (hits.phi_hits<phiMax)]

    #applied the cuts on the number of hits, the pt and remove secondaries
    hitsInfo = hits.merge(nhits, on="particle_id")
    hitsInfo = hitsInfo.merge(InterestingParticles, on = "particle_id")


    #only keep the hit_id and the particle_id
    list_PIDs = list(pd.unique(hitsInfo.particle_id))

    print("============= READ DATA =============")
    print("There are {} particles to reconstruct.".format(len(list_PIDs)))

    return list_PIDs
