from similarity import similarity
import numpy as np
from tree import build_tree_from_list
import bisect
from mol_utils import split_molecule, join_fragments
from global_parameters import MAX_FRAGMENTS
import rdkit.Chem as Chem
import os



# Get a martix containing the similarity of different fragments
def get_dist_matrix(fragments):

    id_dict = {}

    ms = []

    i = 0
    for smi, (m, _) in fragments.items():
        ms.append(m)
        id_dict[i] = smi
        i += 1


    distance_matrix = np.zeros([len(ms)] * 2)

    for i in range(len(ms)):
        for j in range(i+1,len(ms)):
            distance_matrix[i,j] = similarity(id_dict[i], id_dict[j], ms[i], ms[j])
            distance_matrix[j,i] = distance_matrix[i,j]

    return distance_matrix, id_dict




# Create pairs of fragments in a greedy way based on a similarity matrix
def find_pairs(distance_matrix):

    left = np.ones(distance_matrix.shape[0])
    pairs = []

    candidates = sorted(zip(distance_matrix.max(1),zip(range(distance_matrix.shape[0]),
                                                       distance_matrix.argmax(1))))
    use_next = []

    while len(candidates) > 0:
        v, (c1,c2) = candidates.pop()

        if left[c1] + left[c2] == 2:
            left[c1] = 0
            left[c2] = 0
            pairs.append([c1,c2])

        elif np.sum(left) == 1: # Just one sample left
            sampl = np.argmax(left)
            pairs.append([sampl])
            left[sampl] = 0


        elif left[c1] == 1:
            row = distance_matrix[c1,:] * left
            c2_new = row.argmax()
            v_new = row[c2_new]
            new =  (v_new, (c1, c2_new))
            bisect.insort(candidates, new)

    return pairs



# Create a new similarity matrix from a given set of pairs
# The new similarity is the maximal similarity of any fragment in the sets that are combined.
def build_matrix(pairs, old_matrix):

    new_mat = np.zeros([len(pairs)] * 2) - 0.1

    for i in range(len(pairs)):
        for j in range(i+1, len(pairs)):
            new_mat[i,j] = np.max((old_matrix[pairs[i]])[:,[pairs[j]]])
            new_mat[j,i] = new_mat[i,j]
    return new_mat


# Get a containing pairs of nested lists where the similarity between fragments in a list is higher than between
#   fragments which are not in the same list.
def get_hierarchy(fragments):

    distance_matrix,  id_dict = get_dist_matrix(fragments)
    working_mat = (distance_matrix + 0.001) * (1- np.eye(distance_matrix.shape[0]))


    pairings = []

    while working_mat.shape[0] > 1:
        pairings.append(find_pairs(working_mat))
        working_mat = build_matrix(pairings[-1], working_mat)

    return pairings, id_dict



# Build a binary tree from a list of fragments where the most similar fragments are neighbouring in the tree.
# This paths from the root in the tree to the fragments in the leafs is then used to build encode fragments.
def get_encodings(fragments):

    pairings, id_dict = get_hierarchy(fragments)

    assert id_dict

    t = build_tree_from_list(pairings, lookup=id_dict)
    encodings = dict(t.encode_leafs())
    decodings = dict([(v, fragments[k][0]) for k,v in encodings.items()])

    return encodings, decodings



# Encode a fragment.
def encode_molecule(m, encodings):
    fs = [Chem.MolToSmiles(f) for f in split_molecule(m)]
    encoded = "-".join([encodings[f] for f in fs])
    return encoded


# Decode a string representation into a fragment.
def decode_molecule(enc, decodings):
    fs = [Chem.Mol(decodings[x]) for x in enc.split("-")]
    return join_fragments(fs)


# Decode an array representation into a fragment.
def decode(x, translation):
    enc = ["".join([str(int(y)) for y in e[1:]]) for e in x if e[0] == 1]
    fs = [Chem.Mol(translation[e]) for e in enc]
    if not fs:
        return Chem.Mol()
    return join_fragments(fs)


# Encode a list of molecules into their corresponding encodings
def encode_list(mols, encodings):
  
    enc_size = None
    for v in encodings.values():
        enc_size = len(v)
        break
    assert enc_size


    def get_len(x):
        return (len(x) + 1) / enc_size

    encoded_mols = [encode_molecule(m, encodings) for m in mols]
    X_mat = np.zeros((len(encoded_mols), MAX_FRAGMENTS, enc_size + 1))


    for i in range(X_mat.shape[0]):
        es = encoded_mols[i].split("-")

        for j in range(X_mat.shape[1]):
            if j < len(es):
                e = np.asarray([int(c) for c in es[j]])
                if not len(e): continue
                
                X_mat[i,j,0] = 1
                X_mat[i,j,1:] = e

    return X_mat






# Save all decodings as a file (fragments are stored as SMILES)
def save_decodings(decodings):
    decodings_smi = dict([(x,Chem.MolToSmiles(m)) for x,m in decodings.items()])

    if not os.path.exists("History/"):
        os.makedirs("History")

    with open("History/decodings.txt","w+") as f:
        f.write(str(decodings_smi))

# Read encoding list from file
def read_decodings():
    with open("History/decodings.txt","r") as f:
        d = eval(f.read())
        return dict([(x,Chem.MolFromSmiles(m)) for x,m in d.items()])
