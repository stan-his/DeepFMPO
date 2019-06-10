from global_parameters import ETA
import Levenshtein
from rdkit.Chem import rdFMCS


def calculateDistance(smi1,smi2): # calculate edit distance
    return 1 - ETA * Levenshtein.distance(smi1, smi2)


def calculateMCStanimoto(ref_mol, target_mol):

    numAtomsRefCpd = float(ref_mol.GetNumAtoms())
    numAtomsTargetCpd = float(target_mol.GetNumAtoms())

    if numAtomsRefCpd < numAtomsTargetCpd:
        leastNumAtms = int(numAtomsRefCpd)
    else:
        leastNumAtms = int(numAtomsTargetCpd)

    pair_of_molecules = [ref_mol, target_mol]
    numCommonAtoms = rdFMCS.FindMCS(pair_of_molecules, 
                                    atomCompare=rdFMCS.AtomCompare.CompareElements,
                                    bondCompare=rdFMCS.BondCompare.CompareOrderExact, matchValences=True).numAtoms
    mcsTanimoto = numCommonAtoms/((numAtomsTargetCpd+numAtomsRefCpd)-numCommonAtoms)

    return mcsTanimoto, leastNumAtms




def similarity(smi1, smi2, mol1, mol2):
    global s1,s2
    d1 = calculateDistance(smi1, smi2)
    d2 = calculateMCStanimoto(mol1, mol2)[0]
    
    return max(d1, d2)
