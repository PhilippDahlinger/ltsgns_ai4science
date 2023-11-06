import os

import matplotlib
import matplotlib.pyplot as plt
import numpy
import pickle

import tikzplotlib

from scripts.evaluation.plotting_util import plot, adapt

root_path = "/home/philipp/projects/lts_gns/output/evaluations"
experiments = ["lts_gns_prodmp_tissue_manipulation/scalars_C1",
               "lts_gns_prodmp_tissue_manipulation/scalars_C5",
               "lts_gns_prodmp_tissue_manipulation/scalars_C10",
               "mgn_tissue_manipulation/scalars",
               "mgn_poisson_decoder_tissue_manipulation/scalars",
               "mgn_prodmp_tissue_manipulation/scalars",
               ]

names = ["\\gls{ltsgns} C=1",
         "\\gls{ltsgns} C=5",
         "\\gls{ltsgns} C=10",
         "\\gls{mgn}",
         "\\gls{mgn} + Poisson",
         "\\gls{mgn} + \\gls{prodmp}",
         ]

raw_colors = list(matplotlib.colormaps["tab10"].colors)[:len(names)]
colors = [raw_colors[0],
          raw_colors[0],
          raw_colors[0],
          raw_colors[2],
          raw_colors[3],
          raw_colors[4],
          ]

file_name = "tissue_manipulation_mesh_context.tex"
plot(experiments, names, colors, root_path, file_name)
adapt(file_name)
