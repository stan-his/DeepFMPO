import sys
sys.path.insert(0, './Modules/')

from build_encoding import read_decodings, decode
from rewards import evaluate_chem_mol
from rdkit.Chem import Draw
import rdkit.Chem as Chem
import matplotlib.pyplot as plt


import numpy as np



def safe_decode(x, decodings):
    try:
        m = decode(x,decodings)
        Chem.Kekulize(m)
        return m
    except:
        return None
    
    

def main(epoch):
    decodings2 = read_decodings()


    in_mols = np.load("History/in-{}.npy".format(epoch))
    out_mols = np.load("History/out-{}.npy".format(epoch))

    in_mols = [decode(m, decodings2) for m in in_mols]
    out_mols = [safe_decode(m, decodings2) for m in out_mols]

    use = [(not out_mols[i] is None) and \
                  Chem.MolToSmiles(out_mols[i]) != Chem.MolToSmiles(in_mols[i])
           for i in range(len(out_mols))]


    plot_mols = [[m1,m2] for m1,m2,u in zip(in_mols,out_mols,use) if u]
    order = [np.sum(evaluate_chem_mol(out_mols[i])) for i in range(len(out_mols)) if use[i]]
    plot_mols = [x for _,x in sorted(zip(order,plot_mols),reverse=True)]


    plot_mols = [x for y in plot_mols for x in y ]

    # # 
    plot = Draw.MolsToGridImage(plot_mols[:50], molsPerRow=2)
    plot.show()
    
    
if __name__ == "__main__":
    
    if len(sys.argv) > 2:
        epoch = int(sys.argv[1])
        
        main(epoch)