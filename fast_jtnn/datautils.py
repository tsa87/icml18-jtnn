import torch
from torch.utils.data import Dataset, DataLoader
from mol_tree import MolTree
import numpy as np
from jtnn_enc import JTNNEncoder
from mpn import MPN
from jtmpn import JTMPN
import cPickle as pickle
import os, random
from itertools import cycle
import traceback
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
    
class PairTreeFolder(object):

    def __init__(self, data_folder, vocab, batch_size, num_workers=4, shuffle=True, y_assm=True, replicate=None):
        self.data_folder = data_folder
        self.data_files = [fn for fn in os.listdir(data_folder)]
        self.batch_size = batch_size
        self.vocab = vocab
        self.num_workers = num_workers
        self.y_assm = y_assm
        self.shuffle = shuffle

        if replicate is not None: #expand is int
            self.data_files = self.data_files * replicate

    def __iter__(self):
        for fn in self.data_files:
            fn = os.path.join(self.data_folder, fn)
            with open(fn) as f:
                data = pickle.load(f)

            if self.shuffle:
                random.shuffle(data) #shuffle data before batch

            batches = [data[i : i + self.batch_size] for i in xrange(0, len(data), self.batch_size)]
            if len(batches[-1]) < self.batch_size:
                batches.pop()

            dataset = PairTreeDataset(batches, self.vocab, self.y_assm)
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=self.num_workers, collate_fn=lambda x:x[0])

            for b in dataloader:
                yield b

            del data, batches, dataset, dataloader

            
"""Keep track of indices of labeled and unlabeled data we artifically modified for semi-supervised experiment"""
class IndexTracker(object):
    def __init__(self, data_folder, label_idx, label_pct=0.1): 
        self.data_folder = data_folder
        self.data_files = [fn for fn in os.listdir(data_folder)]
        self.unlabeled_idxs_dict = {fn: [] for fn in self.data_files}
        self.labeled_idxs_dict = {fn: [] for fn in self.data_files}
        
        self.label_pct = label_pct
        self.label_idx = label_idx
        
        self.num_pos = 0
        self.num_neg = 0
        self.num_unlabeled = 0
        
        for fn in self.data_files:
            full_fp = os.path.join(self.data_folder, fn) 
            with open(full_fp) as f:
                data = pickle.load(f)
                data = np.array(data).T
            
                labels = data[:, self.label_idx]

                labeled_idxs = np.where(labels != "")[0]
                unlabeled_idxs = np.where(labels == "")[0]
                
                self.num_neg += len(np.where(labels == '0')[0])
                self.num_pos += len(np.where(labels == '1')[0])
                self.num_unlabeled += len(unlabeled_idxs)

                if self.label_pct is not None:
                    labeled_size = len(labeled_idxs)
                    unlabeled_size = len(unlabeled_idxs)
                    
                    target_labeled_size = int(self.label_pct * (labeled_size+unlabeled_size))

                    if labeled_size > target_labeled_size:
                        np.random.shuffle(labeled_idxs)
                        
                        labeled_idxs, newly_unlabeled_idxs = \
                            labeled_idxs[:target_labeled_size], labeled_idxs[target_labeled_size:]
                        
                        unlabeled_idxs = np.append(unlabeled_idxs, newly_unlabeled_idxs)
               
                    else:
                        print "[Warning] No label obscured for " + fn 
                
                self.unlabeled_idxs_dict[fn] = unlabeled_idxs
                self.labeled_idxs_dict[fn] = labeled_idxs

    def get_labeled_idxs(self, fn):
        return self.labeled_idxs_dict[fn]
    
    def get_unlabeled_idxs(self, fn):
        return self.unlabeled_idxs_dict[fn]

    
class MolTreeFolder(object):
    def __init__(
        self, 
        data_folder,
        vocab, 
        batch_size, 
        label_idx, 
        num_workers=4,
        index_tracker=None,
        shuffle=True,
        assm=True,
        replicate=None,
    ):
        
        self.data_folder = data_folder
        self.data_files = [fn for fn in os.listdir(data_folder)]
        self.vocab = vocab
        self.num_workers = num_workers
        self.label_idx = label_idx
        self.index_tracker = index_tracker
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.assm = assm

        if replicate is not None: #expand is int
            self.data_files = self.data_files * replicate
    
    def partition_index(data, shuffle):
        if self.index_tracker is not None:
            labeled_indices = self.index_tracker.get_labeled_idxs(fn)
            unlabeled_indices = self.index_tracker.get_unlabeled_idxs(fn)
        else:
            labeled_indices = np.where(data[:,self.label_idx] != "")[0]
            unlabeled_indices = np.where(data[:,self.label_idx] == "")[0]
        
        if shuffle:
            np.random.shuffle(unlabeled_indices)
            np.random.shuffle(labeled_indices)
          
        return labeled_indices, unlabeled_indices 
            
    def __iter__(self):
        for fn in self.data_files:
            fp = os.path.join(self.data_folder, fn)
            with open(fp) as f:
                data = pickle.load(f)
                data = np.array(data).T
            
            labeled_indices, unlabeled_indices = partition_index(data, self.shuffle)
                        
            supervised_data = data[labeled_indices, :]
            unsupervised_data = data[unlabeled_indices, :]
            del data

            # Create batch
            supervised_moltrees = [supervised_data[i : i + self.batch_size, 0] \
                                  for i in xrange(0, len(supervised_data), self.batch_size)]
            supervised_labels = [supervised_data[i : i + self.batch_size, self.label_idx] \
                                  for i in xrange(0, len(supervised_data), self.batch_size)]
            unsupervised_moltrees = [unsupervised_data[i : i + self.batch_size, 0] \
                                  for i in xrange(0, len(unsupervised_data), self.batch_size)]
            placeholder_labels = [[0 for j in self.batch_size] for i in xrange(0, len(unsupervised_data), self.batch_size)]
            del supervised_data, unsupervised_data
            
            supervised_dataset = MolTreeDataset(supervised_moltree, supervised_labels, self.vocab, self.assm)
            unsupervised_dataset = MolTreeDataset(unsupervised_moltree, placeholder_labels, self.vocab, self.assm)
            del supervised_moltree, supervised_labels, unsupervised_moltree, placeholder_labels 

            supervised_dataloader = DataLoader(
                supervised_dataset, num_workers=self.num_workers, batch_size=1, shuffle=False, collate_fn=lambda x:x[0])
            unsupervised_dataloader = DataLoader(
                unsupervised_dataset, num_workers=self.num_workers, batch_size=1, shuffle=False, collate_fn=lambda x:x[0])
        
            for supervised, unsupervised in zip(cycle(supervised_dataloader), unsupervised_dataloader):
                yield (supervised, unsupervised)
                          
            del supervised_dataset, unsupervised_dataset, unsupervised_dataloader, supervised_dataloader

class PairTreeDataset(Dataset):

    def __init__(self, data, vocab, y_assm):
        self.data = data
        self.vocab = vocab
        self.y_assm = y_assm

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        batch0, batch1 = zip(*self.data[idx])
        return tensorize(batch0, self.vocab, assm=False), tensorize(batch1, self.vocab, assm=self.y_assm)


class MolTreeDataset(Dataset):

    def __init__(self, data, labels, vocab, fp=None, assm=True):
        self.data = data
        self.labels = labels
        self.vocab = vocab
        self.assm = assm

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        try:
            return {
                'data': tensorize(self.data[idx], self.vocab, assm=self.assm),
                'labels': self.labels[idx]
            }
        except Exception as e:
            print idx
            traceback.print_exc()
            

def tensorize(tree_batch, vocab, assm=True):
    set_batch_nodeID(tree_batch, vocab)
    smiles_batch = [tree.smiles for tree in tree_batch]
    jtenc_holder,mess_dict = JTNNEncoder.tensorize(tree_batch)
    jtenc_holder = jtenc_holder
    mpn_holder = MPN.tensorize(smiles_batch)

    if assm is False:
        return tree_batch, jtenc_holder, mpn_holder

    cands = []
    batch_idx = []
    for i,mol_tree in enumerate(tree_batch):
        for node in mol_tree.nodes:
            #Leaf node's attachment is determined by neighboring node's attachment
            if node.is_leaf or len(node.cands) == 1: continue
            cands.extend( [(cand, mol_tree.nodes, node) for cand in node.cands] )
            batch_idx.extend([i] * len(node.cands))

    jtmpn_holder = JTMPN.tensorize(cands, mess_dict)
    batch_idx = torch.LongTensor(batch_idx)

    return tree_batch, jtenc_holder, mpn_holder, (jtmpn_holder,batch_idx)

def set_batch_nodeID(mol_batch, vocab):
    tot = 0
    for mol_tree in mol_batch:
        for node in mol_tree.nodes:
            node.idx = tot
            node.wid = vocab.get_index(node.smiles)
            tot += 1
