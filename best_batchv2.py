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


vol_int = np.array([0, 0, 40])
step = 10
p = ["a", "b", "c"]

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

bt_no =1
node_sel = Node(None, pct_all, None)
while True: 
    max_a = 0
    top_nodes = np.empty(shape=(0), dtype=Node)
    for vol_int in node_sel.pct_nd*min_bottle/100:
        pct_nd = np.array(
            [j for j in node_sel.pct_nd if not np.allclose(j, vol_int * 100 / min_bottle)]
        )
        if node_sel.parent==None:
            root_node = Node(vol_int, pct_nd, 1)
        else:
            root_node = Node(vol_int, pct_nd, 1, node_sel)

        edge_nodes = np.array([root_node])

        for vol in range(min_bottle + 1, max_bottle + 1):
            new_edges = np.empty(shape=(0), dtype=Node)
            for node in edge_nodes:
                if node.pct_nd.size == 0:
                    new_edges = np.append(new_edges, node)
                    continue
                next_vols = node.vol_next(vol)
                if next_vols.size == 0:
                    new_edges = np.append(new_edges, node)
                else:
                    for next_vol in next_vols:
                        new_edges = np.append(
                            new_edges,
                            Node(
                                next_vol,
                                pct_nd_update(node.pct_nd, next_vol * 100 / vol),
                                node.node_count + 1,
                                node,
                            ),
                        )
            edge_nodes = new_edges.copy()

        temp = 0
        for i in edge_nodes:
            if i.node_count>temp:
                temp=i.node_count
                top_node = i

        if temp>max_a:
            max_a = temp
        top_nodes = np.append(top_nodes, top_node)
        print(vol_int)

    # Filter batches with largest number of samples
    temp=np.empty(shape=(0),dtype=Node)
    temp1 = np.empty(shape=(0,max_a,len(p))) # And get history of each top edge-node
    for top_node in top_nodes:
        if top_node.node_count==max_a:
            temp = np.append(temp,top_node)
            temp2=np.array([top_node.curr_vol])
            while top_node.node_count!=1:
                temp2 = np.append([top_node.parent.curr_vol],temp2,axis=0)
                top_node=top_node.parent
            temp1 = np.append(temp1, [temp2], axis=0)
    top_nodes_hist = temp1.copy()
    top_nodes = temp.copy()

    # Select batch with smallest max volumne
    batch_max_vol = max_bottle
    batch_no = 0
    for hist in top_nodes_hist:
        temp = np.sum(hist[-1])
        if temp<batch_max_vol:
            batch_sel = hist
            batch_max_vol = temp
            batch_sel_no = batch_no
        batch_no = batch_no + 1
            
    node_sel = top_nodes[batch_sel_no]
    print(bt_no)
    bt_no = bt_no + 1

    if node_sel.pct_nd.size == 0:
        break

node1 = node_sel
bth_no = bt_no - 1
df = pd.DataFrame(columns = ['batch']+p)
while node1!=None:
    df = df.append(pd.DataFrame([[bth_no] + node1.curr_vol.tolist()], columns = ['batch']+p))
    if node1.node_count==1:
        bth_no = bth_no - 1
    node1 = node1.parent
df.set_index('batch', inplace=True)
df = df[::-1]

# df.to_csv("vol_csv.csv")