import torch.nn as nn

from lts_gns.architectures.util.no_scale_dropout import NoScaleDropout


class GraphDropout(nn.Module):
    def __init__(self, config):
        super(GraphDropout, self).__init__()
        p_node = config.get("node_dropout", 0.0)
        p_edge = config.get("edge_dropout", 0.0)
        p_global = config.get("global_dropout", 0.0)
        self._node_dropout = NoScaleDropout(p=p_node)
        self._edge_dropout = NoScaleDropout(p=p_edge)
        self._global_dropout = NoScaleDropout(p=p_global)

    def forward(self, batch):
        batch.x = self._node_dropout(batch.x)
        batch.edge_attr = self._edge_dropout(batch.edge_attr)
        if hasattr(batch, "u"):
            batch.u = self._global_dropout(batch.u)
        return batch