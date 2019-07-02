import rdkit.Chem as Chem
from rdkit.Chem import Descriptors
import numpy as np
from build_encoding import decode
import rdkit.Chem.Crippen as Crippen
import rdkit.Chem.rdMolDescriptors as MolDescriptors
from rdkit.Chem import Descriptors




# Cache evaluated molecules (rewards are only calculated once)
evaluated_mols = {}




def modify_fragment(f, swap):
    f[-(1+swap)] = (f[-(1+swap)] + 1) % 2
    return f





def get_key(fs):
    return tuple([np.sum([(int(x)* 2 ** (len(a) - y))
                    for x,y in zip(a, range(len(a)))]) if a[0] == 1 \
                     else 0 for a in fs])





# Main function for the evaluation of molecules.
def evaluate_chem_mol(mol):
    try:
        Chem.GetSSSR(mol)
        clogp = Crippen.MolLogP(mol)
        mw = MolDescriptors.CalcExactMolWt(mol)
        tpsa = Descriptors.TPSA(mol)
        ret_val = [
            True,
            320 < mw < 420,
            2 < clogp < 3,
            40 < tpsa < 60
        ]
    except:
        ret_val = [False] * 4

    return ret_val



# Same as above but decodes and check if a cached value could be used.
def evaluate_mol(fs, epoch, decodings):

    global evaluated_mols

    key = get_key(fs)

    if key in evaluated_mols:
        return evaluated_mols[key][0]

    try:
        mol = decode(fs, decodings)
        ret_val = evaluate_chem_mol(mol)
    except:
        ret_val = [False] * 4

    evaluated_mols[key] = (np.array(ret_val), epoch)

    return np.array(ret_val)



# Calculate rewards and give penalty if a locked/empty fragment is changed.
def get_reward(fs,epoch,dist):

    if fs[fs[:,0] == 0].sum() < 0:
        return -0.1

    return (dist * evaluate_mol(fs, epoch)).sum()



# Get initial distribution of rewards among lead molecules
def get_init_dist(X, decodings):

    arr = np.asarray([evaluate_mol(X[i], -1, decodings) for i in range(X.shape[0])])
    dist = arr.shape[0] / (1.0 + arr.sum(0))
    return dist


# Discard molecules which fulfills all targets (used to remove to good lead molecules).
def clean_good(X, decodings):
    X = [X[i] for i in range(X.shape[0]) if not
        evaluate_mol(X[i], -1, decodings).all()]
    return np.asarray(X)

