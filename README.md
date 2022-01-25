# NetworkX based code
A code based on networkx has been developed. Depending on the flag you specify, you either run a walkthrough algorithm or an algorithm based on label component (the latter is still under construction).

This code uses functions that are defined in the tools repository.

filterGraph.py: depending on the arguments, you can either choose to run on solution or prediction.
If you are running on the prediction, then you have to specify the threshold under which you want to remove all the edges.
This code also removes all the isolated nodes

selection.py: this code applies the different selections on the particles, in terms of pT, number of hits. If the graphs are coming from training with a cut on phi or eta, the arguments you set will also take this into account.

plottrackreco.py: this code takes the histos defined in the main function, and fill them according to the variable of interest and the matching criterion. BE AWARE, THE SPLIT CLUSTERS HAVE NOT BEEN TAKEN INTO ACCOUNT YET!

To run the last part of the code and have plots, you should also have a ROOT set up in your environment. You can set it up simply by doing:
```
conda install -c conda-forge root
```

You can either run a walkthrough algorithm or labeling component based algorithm

```
python nx_mywrangler.py --Walk --sol  ## run on solution
python nx_mywrangler.py --Walk --pred --threshold=0.8 ## run on prediction
```

or

```
python nx_mywrangler.py --LabelComp ## under construction, do not look at that
```
