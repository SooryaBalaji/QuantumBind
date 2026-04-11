import requests
from Bio import PDB
import io
import numpy as np

def get_protein_features():
    api_url = "https://alphafold.ebi.ac.uk/api/prediction/P56817"
    response = requests.get(api_url)
    data = response.json()[0]

    pdb_url = data["pdbUrl"]

    pdb_response = requests.get(pdb_url)
    pdb_content = pdb_response.text

    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("BACE1", io.StringIO(pdb_content))

    coords = []
    for model in structure: # 1 protein
        for chain in model: # Different parts of the protein
            for residue in chain: # Amino acids
                for atom in residue: # Actual atoms of amino acids
                    coords.append(atom.get_vector().get_array())

    coords = np.array(coords)
    features = np.array([
        len(coords),  # total atoms
        coords[:, 0].mean(),  # mean x or center coordinates
        coords[:, 1].mean(),  # mean y or center coordinates
        coords[:, 2].mean(),  # mean z or center coordinates
        coords[:, 0].std(),  # std x or how irregular the shape is
        coords[:, 1].std(),  # std y or how irregular the shape is
        coords[:, 2].std(),  # std z or how irregular the shape is
        coords[:, 0].max() - coords[:, 0].min(),  # x span or how tall or wide
        coords[:, 1].max() - coords[:, 1].min(),  # y span or how tall or wide
        coords[:, 2].max() - coords[:, 2].min(),  # z span or how tall or wide
    ], dtype=np.float32)

    return features







