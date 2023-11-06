import os

import matplotlib
import matplotlib.pyplot as plt
import numpy
import pickle

import tikzplotlib

from scripts.evaluation.plotting_util import plot, adapt, plot_last_step

root_path = "/home/philipp/projects/lts_gns/output/evaluations"
# experiments = ["mgn_tissue_manipulation/scalars",
#                "mgn_prodmp_tissue_manipulation/scalars",
#                "lts_gns_prodmp_tissue_manipulation/scalars_C1",
#                "lts_gns_prodmp_tissue_manipulation/scalars_C5",
#                "lts_gns_prodmp_tissue_manipulation/scalars_C10",
#                "lts_gns_prodmp_tissue_manipulation/scalars_C20",
#                "lts_gns_prodmp_tissue_manipulation/scalars_C30",]

experiments = [
    "e2_lts_gns_prodmp_more_blocks_deformable_plate/last_step_eval_C1",
               "e2_lts_gns_prodmp_more_blocks_deformable_plate/last_step_eval_C5",
               "e2_lts_gns_prodmp_more_blocks_deformable_plate/last_step_eval_C10",

               "e2_lts_gns_prodmp_more_blocks_deformable_plate/last_step_eval_PC1",
               "e2_lts_gns_prodmp_more_blocks_deformable_plate/last_step_eval_PC5",
               "e2_lts_gns_prodmp_more_blocks_deformable_plate/last_step_eval_PC10",
               # "e2_lts_gns_prodmp_more_blocks_deformable_plate/last_step_eval_PC20",
               # "e2_lts_gns_prodmp_more_blocks_deformable_plate/last_step_eval_PC30",
               "e2_mgn_deformable_plate/last_step_eval",
               "e2_mgn_poisson_decoder_deformable_plate/last_step_eval",
               "e2_mgn_prodmp_deformable_plate/last_step_eval",
               ]

names = ["C=1",
         "C=5 \\newline \\gls{ltsgns}",
         "C=10",
         "\\gls{ltsgns} PC=1",
         "\\gls{ltsgns} PC=5",
         "\\gls{ltsgns} PC=10",
         # "\\gls{ltsgns} PC=20",
         # "\\gls{ltsgns} PC=30",
         "\\gls{mgn}",
         "\\gls{mgn} + Poisson",
         "\\gls{mgn} + \\gls{prodmp}",
         ]

raw_colors = list(matplotlib.colormaps["tab10"].colors)[:len(names)]
colors = [raw_colors[0],
          raw_colors[0],
          raw_colors[0],
          raw_colors[1],
          raw_colors[1],
          # raw_colors[1],
          # raw_colors[1],
          raw_colors[1],
          raw_colors[2],
          raw_colors[3],
          raw_colors[4],
          ]

file_name = "deformable_plate_last_step.tex"
plot_last_step(experiments, names, colors, root_path, file_name)
adapt(file_name)
