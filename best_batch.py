import numpy as np
import pandas as pd
from itertools import permutations, combinations_with_replacement

# Constants
max_bottle = 150
min_bottle = 40


class Node:
    def __init__(self, curr_vol, pct_nd, node_count, parent=None):
        # pct_nd --> proportions(or)percentages not acheived
        # curr_vol --> current proportions by volume
        # node_count --> no. of nodes till current node
        self.pct_nd = pct_nd
        self.curr_vol = curr_vol
        self.node_count = node_count
        self.parent = parent

    # Return next possible proportions given target volume
    def vol_next(self, vol):
        vol_cand = self.pct_nd * vol / 100
        vol_cand = vol_cand[np.all((vol_cand - self.curr_vol) >= 0, axis=1)]
        return vol_cand


def pct_nd_update(pct_nd, pct_done):
    return np.array([j for j in pct_nd if not np.allclose(j, pct_done)])


vol_int = np.array([0, 40])
step = 10
p = ["a", "b"]

zz = np.array(
    [
        list(list(j) for j in permutations(i))
        for i in combinations_with_replacement(np.arange(0, 101, step), len(p))
        if sum(i) == 100
    ]
)

zz = np.reshape(zz, (-1, len(p)))
zz = np.unique(zz, axis=0)
df = pd.DataFrame(zz, columns=p)

pct_all = df.values

pct_nd = np.array(
    [j for j in pct_all if not np.allclose(j, vol_int * 100 / min_bottle)]
)
root_node = Node(vol_int, pct_nd, 1)
nodes = np.array([root_node])

for vol in range(min_bottle + 1, max_bottle + 1):
    for node in nodes:
        next_vols = node.vol_next(vol)
        for next_vol in next_vols:
            nodes = np.append(
                nodes,
                Node(
                    next_vol,
                    pct_nd_update(node.pct_nd, next_vol * 100 / vol),
                    node.node_count + 1,
                    node,
                ),
            )
    print(vol)

a = 0
for i in nodes:
    if i.node_count > a:
        z = node

aa = []
for i in nodes:
    if i.node_count == 3:
        aa.append(i)

