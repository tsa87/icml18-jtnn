import os

import torch
import torch.nn as nn
from multiprocessing import Pool

import math, random, sys
import numpy as np
from optparse import OptionParser
import pickle

from fast_jtnn import *
import rdkit

from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem

def tensorize(smiles, assm=True):
    mol_tree = MolTree(smiles)
    mol_tree.recover()
    if assm:
        mol_tree.assemble()
        for node in mol_tree.nodes:
            if node.label not in node.cands:
                node.cands.append(node.label)

    del mol_tree.mol
    for node in mol_tree.nodes:
        del node.mol

    return mol_tree


def get_fp(smiles, n_bits=1024):
    ms = [Chem.MolFromSmiles(smile) for smile in smiles]
    fingerprints = [AllChem.GetHashedMorganFingerprint(m, 2, n_bits) for m in ms]
    np_arr = np.zeros((len(smiles),n_bits), dtype=np.float32)
    for i, fingerprint in enumerate(fingerprints):
        DataStructs.ConvertToNumpyArray(fingerprint, np_arr[i]) 

    return np_arr


if __name__ == "__main__":
    import csv
    from tqdm import tqdm

    lg = rdkit.RDLogger.logger()
    lg.setLevel(rdkit.RDLogger.CRITICAL)

    parser = OptionParser()
    parser.add_option("-t", "--train", dest="train_path", help="csv file containing smiles columns and label columns.")
    parser.add_option("-s", "--save", dest="save_path")
    parser.add_option("-d", "--data_col", dest="data_col", type=int)
    parser.add_option("-n", "--split", dest="nsplits", default=10)
    parser.add_option("-j", "--jobs", dest="njobs", default=8)
    opts,args = parser.parse_args()
    opts.njobs = int(opts.njobs)

    pool = Pool(opts.njobs)
    num_splits = int(opts.nsplits)

    reader = csv.reader(open(opts.train_path, 'r'), delimiter=',')
    next(reader)
    data = list(reader)
    data = map(list, zip(*data))

    labels = [x for x in range(len(data)) if x != opts.data_col]

    smiles = data[opts.data_col]
    fps =  get_fp(smiles)
    print fps
    all_labels = [data[c] for c in labels]
    all_mol_tree = list(tqdm(pool.imap(tensorize, smiles), total=len(smiles)))

    le = (len(all_mol_tree) + num_splits - 1) / num_splits

    for split_id in xrange(num_splits):
        st = split_id * le
        sub_data = [smiles[st : st + le]] + [all_mol_tree[st : st + le]] + [fps[st : st + le]] + [label[st : st + le] for label in all_labels]
        with open('tensors-%d.pkl' % split_id, 'wb') as f:
            pickle.dump(sub_data, f, pickle.HIGHEST_PROTOCOL)
