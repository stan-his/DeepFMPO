import sys
sys.path.insert(0, './Modules/')

import numpy as np

from file_reader import read_file
from mol_utils import get_fragments
from build_encoding import get_encodings, encode_molecule, decode_molecule, encode_list, save_decodings
from models import build_models
from training import train
from rewards import clean_good
from rdkit import rdBase
rdBase.DisableLog('rdApp.error')


def main(fragment_file, lead_file):
    fragment_mols = read_file(fragment_file)
    lead_mols = read_file(lead_file)
    fragment_mols += lead_mols


    fragments, used_mols = get_fragments(fragment_mols)
    encodings, decodings = get_encodings(fragments)
    save_decodings(decodings)

    lead_mols = np.asarray(fragment_mols[-len(lead_mols):])[used_mols[-len(lead_mols):]]

    X = encode_list(lead_mols, encodings)

    actor, critic = build_models(X.shape[1:])

    X = clean_good(X, decodings)

    history = train(X, actor, critic, decodings)

    np.save("History/history.npy", history)

    
    
    
if __name__ == "__main__":
    
    fragment_file = "../molecules.smi"
    lead_file = "../dopamineD4props.csv"

    if len(sys.argv) > 1:
        fragment_file = sys.argv[1]
        
    if len(sys.argv) > 2:
        lead_file = sys.argv[2]
        
    main(fragment_file, lead_file)
        
