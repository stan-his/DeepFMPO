from global_parameters import ETA
import Levenshtein
from rdkit.Chem import rdFMCS



# Calculate similartity between two molecules (or fragments) based on their edit distance
def calculateDistance(smi1,smi2): 
    return 1 - ETA * Levenshtein.distance(smi1, smi2)


# Calculate the MCS Tanimoto similarity between two molecules
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



# Calculate the similarity of two molecules (with SMILE representations smi1 and smi2) 
#  This is the maximum of the two functions above
def similarity(smi1, smi2, mol1, mol2):
    global s1,s2
    d1 = calculateDistance(smi1, smi2)
    d2 = calculateMCStanimoto(mol1, mol2)[0]
    
    return max(d1, d2)
