import os

import matplotlib
import matplotlib.pyplot as plt
import numpy
import pickle

import tikzplotlib

from scripts.evaluation.plotting_util import plot, adapt, plot_last_step

root_path = "/home/philipp/projects/lts_gns/output/evaluations"
experiments = ["lts_gns_prodmp_tissue_manipulation/last_step_eval_C1",
               "lts_gns_prodmp_tissue_manipulation/last_step_eval_C5",
               "lts_gns_prodmp_tissue_manipulation/last_step_eval_C10",
               # "lts_gns_prodmp_tissue_manipulation/last_step_eval_C20",
               # "lts_gns_prodmp_tissue_manipulation/last_step_eval_C30",
               "mgn_tissue_manipulation/last_step_eval",
               "mgn_poisson_decoder_tissue_manipulation/last_step_eval",
               "mgn_prodmp_tissue_manipulation/last_step_eval",
               ]

names = ["\\gls{ltsgns} C=1",
         "\\gls{ltsgns} C=5",
         "\\gls{ltsgns} C=10",
         "\\gls{mgn}",
         "\\gls{mgn} + Poisson",
         # "\\gls{mgn} + Poisson",
         # "\\gls{mgn} + Poisson",
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

file_name = "tissue_manipulation_last_step.tex"
plot_last_step(experiments, names, colors, root_path, file_name)
adapt(file_name)
