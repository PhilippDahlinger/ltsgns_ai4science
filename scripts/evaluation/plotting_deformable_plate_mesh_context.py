import os

import matplotlib
import matplotlib.pyplot as plt
import numpy
import pickle

import tikzplotlib

from scripts.evaluation.plotting_util import plot, adapt

root_path = "/home/philipp/projects/lts_gns/output/evaluations"
# experiments = ["mgn_tissue_manipulation/scalars",
#                "mgn_prodmp_tissue_manipulation/scalars",
#                "lts_gns_prodmp_tissue_manipulation/scalars_C1",
#                "lts_gns_prodmp_tissue_manipulation/scalars_C5",
#                "lts_gns_prodmp_tissue_manipulation/scalars_C10",
#                "lts_gns_prodmp_tissue_manipulation/scalars_C20",
#                "lts_gns_prodmp_tissue_manipulation/scalars_C30",]

experiments = ["e2_lts_gns_prodmp_more_blocks_deformable_plate/scalars_C1",
               "e2_lts_gns_prodmp_more_blocks_deformable_plate/scalars_C5",
               "e2_lts_gns_prodmp_more_blocks_deformable_plate/scalars_C10",

               "e2_lts_gns_prodmp_more_blocks_deformable_plate/scalars_PC1",
               "e2_lts_gns_prodmp_more_blocks_deformable_plate/scalars_PC5",
               "e2_lts_gns_prodmp_more_blocks_deformable_plate/scalars_PC10",
               "e2_mgn_deformable_plate/scalars",
               "e2_mgn_poisson_decoder_deformable_plate/scalars",
               "e2_mgn_prodmp_deformable_plate/scalars",
               ]

names = ["C=1",
         "C=5 \\newline \\gls{ltsgns}",
         "C=10",
         "\\gls{ltsgns} PC=1",
         "\\gls{ltsgns} PC=5",
         "\\gls{ltsgns} PC=10",
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
          raw_colors[1],
          raw_colors[2],
          raw_colors[3],
          raw_colors[4],
          ]

file_name = "deformable_plate.tex"
plot(experiments, names, colors, root_path, file_name)
adapt(file_name)
