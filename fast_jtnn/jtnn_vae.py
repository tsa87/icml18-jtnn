import torch
import torch.nn as nn
import torch.nn.functional as F
from mol_tree import Vocab, MolTree
from nnutils import create_var, flatten_tensor, avg_pool, create_onehot, log_standard_categorical
from jtnn_enc import JTNNEncoder
from jtnn_dec import JTNNDecoder
from mpn import MPN
from jtmpn import JTMPN
from datautils import tensorize


from chemutils import enum_assemble, set_atommap, copy_edit_mol, attach_mols
import rdkit
import rdkit.Chem as Chem
import copy, math
from itertools import chain

import numpy as np

import sys


class JTNNVAE(nn.Module):

    def __init__(self, vocab, hidden_size, latent_size, depthT, depthG):
        super(JTNNVAE, self).__init__()
        self.vocab = vocab
        self.hidden_size = hidden_size
        self.latent_size = latent_size = latent_size / 2 #Tree and Mol has two vectors

        self.jtnn = JTNNEncoder(hidden_size, depthT, nn.Embedding(vocab.size(), hidden_size))
        self.decoder = JTNNDecoder(vocab, hidden_size, latent_size, nn.Embedding(vocab.size(), hidden_size))

        self.jtmpn = JTMPN(hidden_size, depthG)
        self.mpn = MPN(hidden_size, depthG)

        self.A_assm = nn.Linear(latent_size, hidden_size, bias=False)
        self.assm_loss = nn.CrossEntropyLoss(size_average=False)

        self.T_mean = nn.Linear(hidden_size, latent_size)
        self.T_var = nn.Linear(hidden_size, latent_size)
        self.G_mean = nn.Linear(hidden_size, latent_size)
        self.G_var = nn.Linear(hidden_size, latent_size)

    def encode(self, jtenc_holder, mpn_holder):
        tree_vecs, tree_mess = self.jtnn(*jtenc_holder)
        mol_vecs = self.mpn(*mpn_holder)
        return tree_vecs, tree_mess, mol_vecs
    
    def encode_from_smiles(self, smiles_list):
        tree_batch = [MolTree(s) for s in smiles_list]
        _, jtenc_holder, mpn_holder = tensorize(tree_batch, self.vocab, assm=False)
        tree_vecs, _, mol_vecs = self.encode(jtenc_holder, mpn_holder)
        return torch.cat([tree_vecs, mol_vecs], dim=-1)

    def encode_latent(self, jtenc_holder, mpn_holder):
        tree_vecs, _ = self.jtnn(*jtenc_holder)
        mol_vecs = self.mpn(*mpn_holder)
        tree_mean = self.T_mean(tree_vecs)
        mol_mean = self.G_mean(mol_vecs)
        tree_var = -torch.abs(self.T_var(tree_vecs))
        mol_var = -torch.abs(self.G_var(mol_vecs))
        return torch.cat([tree_mean, mol_mean], dim=1), torch.cat([tree_var, mol_var], dim=1)

    def rsample(self, z_vecs, W_mean, W_var):
        batch_size = z_vecs.size(0)
        z_mean = W_mean(z_vecs)
        z_log_var = -torch.abs(W_var(z_vecs)) #Following Mueller et al.
        kl_loss = -0.5 * torch.sum(1.0 + z_log_var - z_mean * z_mean - torch.exp(z_log_var)) / batch_size
        epsilon = create_var(torch.randn_like(z_mean))
        z_vecs = z_mean + torch.exp(z_log_var / 2) * epsilon
        return z_vecs, kl_loss

    def sample_prior(self, prob_decode=False):
        z_tree = torch.randn(1, self.latent_size).cuda()
        z_mol = torch.randn(1, self.latent_size).cuda()
        return self.decode(z_tree, z_mol, prob_decode)

    def forward(self, x_batch, beta):
        x_batch, x_jtenc_holder, x_mpn_holder, x_jtmpn_holder = x_batch
        x_tree_vecs, x_tree_mess, x_mol_vecs = self.encode(x_jtenc_holder, x_mpn_holder)
        z_tree_vecs,tree_kl = self.rsample(x_tree_vecs, self.T_mean, self.T_var)
        z_mol_vecs,mol_kl = self.rsample(x_mol_vecs, self.G_mean, self.G_var)

        kl_div = tree_kl + mol_kl
        word_loss, topo_loss, word_acc, topo_acc = self.decoder(x_batch, z_tree_vecs)
        assm_loss, assm_acc = self.assm(x_batch, x_jtmpn_holder, z_mol_vecs, x_tree_mess)

        return word_loss + topo_loss + assm_loss + beta * kl_div, kl_div.item(), word_acc, topo_acc, assm_acc

    def assm(self, mol_batch, jtmpn_holder, x_mol_vecs, x_tree_mess):
        jtmpn_holder,batch_idx = jtmpn_holder
        fatoms,fbonds,agraph,bgraph,scope = jtmpn_holder
        batch_idx = create_var(batch_idx)

        cand_vecs = self.jtmpn(fatoms, fbonds, agraph, bgraph, scope, x_tree_mess)

        x_mol_vecs = x_mol_vecs.index_select(0, batch_idx)
        x_mol_vecs = self.A_assm(x_mol_vecs) #bilinear
        scores = torch.bmm(
                x_mol_vecs.unsqueeze(1),
                cand_vecs.unsqueeze(-1)
        ).squeeze()
        
        cnt,tot,acc = 0,0,0
        all_loss = []
        for i,mol_tree in enumerate(mol_batch):
            comp_nodes = [node for node in mol_tree.nodes if len(node.cands) > 1 and not node.is_leaf]
            cnt += len(comp_nodes)
            for node in comp_nodes:
                label = node.cands.index(node.label)
                ncand = len(node.cands)
                cur_score = scores.narrow(0, tot, ncand)
                tot += ncand

                if cur_score.data[label] >= cur_score.max().item():
                    acc += 1

                label = create_var(torch.LongTensor([label]))
                all_loss.append( self.assm_loss(cur_score.view(1,-1), label) )
        
        all_loss = sum(all_loss) / len(mol_batch)
        return all_loss, acc * 1.0 / cnt

    def decode(self, x_tree_vecs, x_mol_vecs, prob_decode):
        #currently do not support batch decoding
        assert x_tree_vecs.size(0) == 1 and x_mol_vecs.size(0) == 1

        pred_root,pred_nodes = self.decoder.decode(x_tree_vecs, prob_decode)
        if len(pred_nodes) == 0: return None
        elif len(pred_nodes) == 1: return pred_root.smiles

        #Mark nid & is_leaf & atommap
        for i,node in enumerate(pred_nodes):
            node.nid = i + 1
            node.is_leaf = (len(node.neighbors) == 1)
            if len(node.neighbors) > 1:
                set_atommap(node.mol, node.nid)

        scope = [(0, len(pred_nodes))]
        jtenc_holder,mess_dict = JTNNEncoder.tensorize_nodes(pred_nodes, scope)
        _,tree_mess = self.jtnn(*jtenc_holder)
        tree_mess = (tree_mess, mess_dict) #Important: tree_mess is a matrix, mess_dict is a python dict

        x_mol_vecs = self.A_assm(x_mol_vecs).squeeze() #bilinear

        cur_mol = copy_edit_mol(pred_root.mol)
        global_amap = [{}] + [{} for node in pred_nodes]
        global_amap[1] = {atom.GetIdx():atom.GetIdx() for atom in cur_mol.GetAtoms()}

        cur_mol,_ = self.dfs_assemble(tree_mess, x_mol_vecs, pred_nodes, cur_mol, global_amap, [], pred_root, None, prob_decode, check_aroma=True)
        if cur_mol is None: 
            cur_mol = copy_edit_mol(pred_root.mol)
            global_amap = [{}] + [{} for node in pred_nodes]
            global_amap[1] = {atom.GetIdx():atom.GetIdx() for atom in cur_mol.GetAtoms()}
            cur_mol,pre_mol = self.dfs_assemble(tree_mess, x_mol_vecs, pred_nodes, cur_mol, global_amap, [], pred_root, None, prob_decode, check_aroma=False)
            if cur_mol is None: cur_mol = pre_mol

        if cur_mol is None: 
            return None

        cur_mol = cur_mol.GetMol()
        set_atommap(cur_mol)
        cur_mol = Chem.MolFromSmiles(Chem.MolToSmiles(cur_mol))
        return Chem.MolToSmiles(cur_mol) if cur_mol is not None else None
        
    def dfs_assemble(self, y_tree_mess, x_mol_vecs, all_nodes, cur_mol, global_amap, fa_amap, cur_node, fa_node, prob_decode, check_aroma):
        fa_nid = fa_node.nid if fa_node is not None else -1
        prev_nodes = [fa_node] if fa_node is not None else []

        children = [nei for nei in cur_node.neighbors if nei.nid != fa_nid]
        neighbors = [nei for nei in children if nei.mol.GetNumAtoms() > 1]
        neighbors = sorted(neighbors, key=lambda x:x.mol.GetNumAtoms(), reverse=True)
        singletons = [nei for nei in children if nei.mol.GetNumAtoms() == 1]
        neighbors = singletons + neighbors

        cur_amap = [(fa_nid,a2,a1) for nid,a1,a2 in fa_amap if nid == cur_node.nid]
        cands,aroma_score = enum_assemble(cur_node, neighbors, prev_nodes, cur_amap)
        if len(cands) == 0 or (sum(aroma_score) < 0 and check_aroma):
            return None, cur_mol

        cand_smiles,cand_amap = zip(*cands)
        aroma_score = torch.Tensor(aroma_score).cuda()
        cands = [(smiles, all_nodes, cur_node) for smiles in cand_smiles]

        if len(cands) > 1:
            jtmpn_holder = JTMPN.tensorize(cands, y_tree_mess[1])
            fatoms,fbonds,agraph,bgraph,scope = jtmpn_holder
            cand_vecs = self.jtmpn(fatoms, fbonds, agraph, bgraph, scope, y_tree_mess[0])
            scores = torch.mv(cand_vecs, x_mol_vecs) + aroma_score
        else:
            scores = torch.Tensor([1.0])

        if prob_decode:
            probs = F.softmax(scores.view(1,-1), dim=1).squeeze() + 1e-7 #prevent prob = 0
            cand_idx = torch.multinomial(probs, probs.numel())
        else:
            _,cand_idx = torch.sort(scores, descending=True)

        backup_mol = Chem.RWMol(cur_mol)
        pre_mol = cur_mol
        for i in xrange(cand_idx.numel()):
            cur_mol = Chem.RWMol(backup_mol)
            pred_amap = cand_amap[cand_idx[i].item()]
            new_global_amap = copy.deepcopy(global_amap)

            for nei_id,ctr_atom,nei_atom in pred_amap:
                if nei_id == fa_nid:
                    continue
                new_global_amap[nei_id][nei_atom] = new_global_amap[cur_node.nid][ctr_atom]

            cur_mol = attach_mols(cur_mol, children, [], new_global_amap) #father is already attached
            new_mol = cur_mol.GetMol()
            new_mol = Chem.MolFromSmiles(Chem.MolToSmiles(new_mol))

            if new_mol is None: continue
            
            has_error = False
            for nei_node in children:
                if nei_node.is_leaf: continue
                tmp_mol, tmp_mol2 = self.dfs_assemble(y_tree_mess, x_mol_vecs, all_nodes, cur_mol, new_global_amap, pred_amap, nei_node, cur_node, prob_decode, check_aroma)
                if tmp_mol is None: 
                    has_error = True
                    if i == 0: pre_mol = tmp_mol2
                    break
                cur_mol = tmp_mol

            if not has_error: return cur_mol, cur_mol

        return None, pre_mol

    
class SemiJTNNVAE(JTNNVAE):
    def __init__(self, vocab, hidden_size, latent_size, y_size, depthT, depthG, alpha):
        super(SemiJTNNVAE, self).__init__(vocab, hidden_size, latent_size, depthT, depthG)
        
        self.y_size = y_size
        self.alpha = alpha
    
        self.predictor = nn.Sequential(
            nn.Linear(hidden_size*2, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, y_size))
        
        latent_size /= 2
        self.T_mean = nn.Linear(hidden_size+y_size, latent_size)
        self.T_var = nn.Linear(hidden_size+y_size, latent_size)
        self.G_mean = nn.Linear(hidden_size+y_size, latent_size)
        self.G_var = nn.Linear(hidden_size+y_size, latent_size)
        
    def rsample(self, z_vecs, y_hat, W_mean, W_var):
        z_vecs = torch.cat((z_vecs, y_hat), 1)
        batch_size = z_vecs.size(0)
        z_mean = W_mean(z_vecs)
        z_log_var = -torch.abs(W_var(z_vecs)) #Following Mueller et al.
        kl_loss = -0.5 * torch.sum(1.0 + z_log_var - z_mean * z_mean - torch.exp(z_log_var)) / batch_size
        epsilon = create_var(torch.randn_like(z_mean))
        z_vecs = z_mean + torch.exp(z_log_var / 2) * epsilon
        return z_vecs, kl_loss
    
    def predict(self, x_batch):
        x_batch, x_jtenc_holder, x_mpn_holder, x_jtmpn_holder = x_batch  
        x_tree_vecs, x_tree_mess, x_mol_vecs = self.encode(x_jtenc_holder, x_mpn_holder)

        y_hat = self.predictor(torch.cat((x_tree_vecs, x_mol_vecs), 1))  
        return y_hat
    
    # L(x,y) without log(y)
    def _compute_partial_loss(self, y_batch, x_batch, x_tree_vecs, x_tree_mess, x_mol_vecs, x_jtmpn_holder, beta):
        z_tree_vecs, tree_kl = self.rsample(x_tree_vecs, y_batch, self.T_mean, self.T_var)
        z_mol_vecs, mol_kl = self.rsample(x_mol_vecs, y_batch, self.G_mean, self.G_var)

        kl_div = tree_kl + mol_kl

        word_loss, topo_loss, word_acc, topo_acc = self.decoder(x_batch, z_tree_vecs)
        assm_loss, assm_acc = self.assm(x_batch, x_jtmpn_holder, z_mol_vecs, x_tree_mess)

        L_xy = word_loss + topo_loss + assm_loss + beta * kl_div
        return L_xy, kl_div, word_acc, topo_acc, assm_acc
    
class SemiJTNNVAEClassifier(SemiJTNNVAE):

    def __init__(self, vocab, hidden_size, latent_size, y_size, depthT, depthG, alpha, weight=None):
        super(SemiJTNNVAEClassifier, self).__init__(vocab, hidden_size, latent_size, y_size, depthT, depthG, alpha)

        self.predictor = nn.Sequential(
            nn.Linear(hidden_size*2, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, y_size),
            nn.Softmax()
        )
        
        if weight is not None:
            self.pred_loss = nn.CrossEntropyLoss(weight=weight, reduction='none')
        else:
            self.pred_loss = nn.CrossEntropyLoss()
                
    def classify(self, x_batch):
        y_hat = self.predict(x_batch)    
        return torch.argmax(y_hat, axis=1)
        
    def forward(self, x_batch, y_batch, beta):
        is_labeled = False if y_batch is None else True

        x_batch, x_jtenc_holder, x_mpn_holder, x_jtmpn_holder = x_batch  
        x_tree_vecs, x_tree_mess, x_mol_vecs = self.encode(x_jtenc_holder, x_mpn_holder)

        y_hat = self.predictor(torch.cat((x_tree_vecs, x_mol_vecs), 1))
        loss, kl_div, word_acc, topo_acc, assm_acc, clsf_acc, clsf_loss = 0, 0, 0, 0, 0, 0, 0
        
        pred = None
        target = None
        
        if not is_labeled:
            # Eq(y|x)[-L(x,y)]
            for i in range(self.y_size):
                y_batch = create_onehot(len(x_batch), self.y_size, i)
                    
                L_xy, kl, wacc, tacc, aacc = \
                    self._compute_partial_loss(y_batch, x_batch, x_tree_vecs, x_tree_mess, x_mol_vecs, x_jtmpn_holder, beta)
                
                kl_div += kl / self.y_size
                word_acc += wacc / self.y_size
                topo_acc += tacc / self.y_size
                assm_acc += aacc / self.y_size
                
                loss += L_xy * torch.mean(y_hat[:, i])

            # H(q(y|x)
            y_hat_entropy = torch.sum(y_hat * torch.log(y_hat + 1e-8))
            loss += y_hat_entropy

        else:
            y_batch = create_var(y_batch)

            L_xy, kl_div, word_acc, topo_acc, assm_acc = \
                self._compute_partial_loss(y_batch, x_batch, x_tree_vecs, x_tree_mess, x_mol_vecs, x_jtmpn_holder, beta)

            target = torch.argmax(y_batch, axis=1)
            pred = torch.argmax(y_hat, axis=1)
            
            clsf_loss = torch.mean(self.pred_loss(y_hat, target))           
            clsf_acc = float((target == pred).sum().item()) / pred.size(0)

            loss += L_xy + clsf_loss * self.alpha
            
        logy = torch.mean(log_standard_categorical(y_hat))
        loss += logy
        
        return loss, clsf_loss*self.alpha, clsf_acc, (pred, target)
    
    
class SemiJTNNVAERegressor(SemiJTNNVAE):

    def __init__(self, vocab, hidden_size, latent_size, y_size, depthT, depthG, alpha):
        super(SemiJTNNVAERegressor, self).__init__(vocab, hidden_size, latent_size, y_size, depthT, depthG, alpha)
        
        self.predictor = nn.Sequential(
            nn.Linear(hidden_size*2, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, y_size*2))
        
        self.pred_loss = nn.MSELoss()

            
    def forward(self, x_batch, y_batch, beta):
        is_labeled = False if y_batch is None else True

        x_batch, x_jtenc_holder, x_mpn_holder, x_jtmpn_holder = x_batch  
        x_tree_vecs, x_tree_mess, x_mol_vecs = self.encode(x_jtenc_holder, x_mpn_holder)

        y_mean, y_std = torch.chunk(self.predictor(torch.cat((x_tree_vecs, x_mol_vecs), 1)), 2, dim=1)
        loss, kl_div, word_acc, topo_acc, assm_acc, clsf_acc, clsf_loss = 0, 0, 0, 0, 0, 0, 0
        
        pred = None
        target = None
        
        if not is_labeled:
            # Eq(y|x)[-L(x,y)]
            y_batch = torch.normal(mean=y_mean, std=y_std)
            
            L_xy, kl, wacc, tacc, aacc = \
                self._compute_partial_loss(y_batch, x_batch, x_tree_vecs, x_tree_mess, x_mol_vecs, x_jtmpn_holder, beta)

            kl_div += kl / self.y_size
            word_acc += wacc / self.y_size
            topo_acc += tacc / self.y_size
            assm_acc += aacc / self.y_size

            # H(q(y|x))
            y_hat_entropy = torch.sum(y_batch* torch.log(y_batch + 1e-8))
            loss = L_xy + 2*y_hat_entropy
        else:
            y_batch = create_var(y_batch)

            L_xy, kl_div, word_acc, topo_acc, assm_acc = \
                self._compute_partial_loss(y_batch, x_batch, x_tree_vecs, x_tree_mess, x_mol_vecs, x_jtmpn_holder, beta)
            
            pred_loss = torch.mean(self.pred_loss(y_hat, y_batch))           
            loss += L_xy + pred_loss * self.alpha
            
        # loss + log(y)
        return loss, pred_loss*self.alpha, (pred, target)