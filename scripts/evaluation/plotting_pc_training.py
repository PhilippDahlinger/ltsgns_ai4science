import os

import matplotlib
import matplotlib.pyplot as plt
import numpy
import pickle

import tikzplotlib

from scripts.evaluation.plotting_util import plot, adapt

root_path = "/home/philipp/projects/lts_gns/output/evaluations"

experiments = ["e2_lts_gns_prodmp_more_blocks_deformable_plate/scalars_PC1",
               "e2_lts_gns_prodmp_more_blocks_deformable_plate/scalars_PC5",
               "e2_lts_gns_prodmp_more_blocks_deformable_plate/scalars_PC10",
               "e2_lts_gns_prodmp_more_blocks_deformable_plate/scalars_PC20",
               "e2_lts_gns_prodmp_pc_training_deformable_plate/scalars_PC1",
               "e2_lts_gns_prodmp_pc_training_deformable_plate/scalars_PC5",
               "e2_lts_gns_prodmp_pc_training_deformable_plate/scalars_PC10",
               "e2_lts_gns_prodmp_pc_training_deformable_plate/scalars_PC20",
               "e2_mgn_deformable_plate/scalars",
               "e2_mgn_poisson_decoder_deformable_plate/scalars",
               ]

names = ["\\gls{ltsgns} ($|\\mathcal{D}_*^c|=1$)",
         "\\gls{ltsgns} ($|\\mathcal{D}_*^c|=5$)",
         "\\gls{ltsgns} ($|\\mathcal{D}_*^c|=10$)",
         "\\gls{ltsgns} ($|\\mathcal{D}_*^c|=20$)",
         "\\gls{ltsgns} PC train ($|\\mathcal{D}_*^c|=1$)",
         "\\gls{ltsgns} PC train ($|\\mathcal{D}_*^c|=5$)",
         "\\gls{ltsgns} PC train ($|\\mathcal{D}_*^c|=10$)",
         "\\gls{ltsgns} PC train ($|\\mathcal{D}_*^c|=20$)",
         "\\gls{mgn}",
         "\\gls{mgn} + Poisson",
         ]

colors = list(matplotlib.colormaps["tab10"].colors)[:len(names)]

file_name = "pc_training.tex"
plot(experiments, names, colors, root_path, file_name)
adapt(file_name)
