from Bio import Phylo, SeqIO
from Bio.Phylo.TreeConstruction import DistanceTreeConstructor, DistanceMatrix
import numpy as np
import os, argparse, sys
import pandas as pd

def parse_input_matrix(input_file):
    _, ext = os.path.splitext(input_file.lower())
    taxa = []
    sequences = []

    if ext in ['.fasta', '.fa', '.fst', '.matrix']:
        for record in SeqIO.parse(input_file, "fasta"):
            taxa.append(record.id)
            seq = [int(char) if char.isdigit() else char for char in str(record.seq)]
            sequences.append(seq)
    elif ext == '.csv':
        df = pd.read_csv(input_file, index_col=0)
        taxa = df.index.tolist()
        sequences = df.values.tolist()
    else:
        raise ValueError(f"Unsupported file format: {ext}")

    # Check that all sequences are of equal length
    lengths = [len(seq) for seq in sequences]
    if len(set(lengths)) > 1:
        raise ValueError(f"Inconsistent number of columns across rows in file {input_file}. Found lengths: {set(lengths)}")

    return taxa, np.array(sequences)

def build_tree(taxa, matrix):
    num_taxa = len(taxa)
    dist_matrix = []
    for i in range(num_taxa):
        row = []
        for j in range(i + 1):
            dist = np.sum(matrix[i] != matrix[j])
            row.append(dist)
        dist_matrix.append(row)
    distance_matrix = DistanceMatrix(names=taxa, matrix=dist_matrix)
    constructor = DistanceTreeConstructor()
    return constructor.upgma(distance_matrix)

# Parsimony algorithms: Camin-Sokal, Dollo, Wagner, Fitch
def camin_sokal_parsimony(tree, taxa, matrix):
    taxon_to_seq = {taxa[i]: matrix[i] for i in range(len(taxa))}
    node_states = {leaf: taxon_to_seq[leaf.name] for leaf in tree.get_terminals()}

    for node in tree.get_nonterminals(order='postorder'):
        child_states = [node_states[child] for child in node.clades]
        node_states[node] = [0 if all(c == 0 for c in chars) else 1 for chars in zip(*child_states)]

    score = 0
    for node in tree.get_nonterminals(order='preorder'):
        for child in node.clades:
            score += sum(1 for p, c in zip(node_states[node], node_states[child]) if p == 0 and c == 1)
    return score, node_states

def dollo_parsimony(tree, taxa, matrix):
    taxon_to_seq = {taxa[i]: matrix[i] for i in range(len(taxa))}
    node_states = {leaf: taxon_to_seq[leaf.name] for leaf in tree.get_terminals()}

    for node in tree.get_nonterminals(order='postorder'):
        child_states = [node_states[child] for child in node.clades]
        node_states[node] = [1 if any(c == 1 for c in chars) else 0 for chars in zip(*child_states)]

    site_gains = [False] * len(matrix[0])
    score = 0
    for node in tree.get_nonterminals(order='preorder'):
        for child in node.clades:
            for i, (p, c) in enumerate(zip(node_states[node], node_states[child])):
                if p == 0 and c == 1 and not site_gains[i]:
                    site_gains[i] = True
                    score += 1
                elif p == 1 and c == 0:
                    score += 1
    return score, node_states

def wagner_parsimony(tree, taxa, matrix):
    taxon_to_seq = {taxa[i]: matrix[i] for i in range(len(taxa))}
    node_states = {leaf: taxon_to_seq[leaf.name] for leaf in tree.get_terminals()}

    for node in tree.get_nonterminals(order='postorder'):
        child_states = [node_states[child] for child in node.clades]
        node_states[node] = [a if a == b else 1 for a, b in zip(*child_states)]

    score = 0
    for node in tree.get_nonterminals(order='preorder'):
        for child in node.clades:
            score += sum(1 for p, c in zip(node_states[node], node_states[child]) if p != c)
    return score, node_states

def fitch_parsimony(tree, taxa, matrix):
    taxon_to_seq = {taxa[i]: matrix[i] for i in range(len(taxa))}
    node_states = {leaf: [{c} for c in taxon_to_seq[leaf.name]] for leaf in tree.get_terminals()}

    for node in tree.get_nonterminals(order='postorder'):
        s1, s2 = node.clades
        merged = []
        for a, b in zip(node_states[s1], node_states[s2]):
            intersect = a & b
            merged.append(intersect if intersect else a | b)
        node_states[node] = merged

    score = 0
    for node in tree.get_nonterminals(order='preorder'):
        for child in node.clades:
            score += sum(1 for a, b in zip(node_states[node], node_states[child]) if not a & b)
    return score, node_states

def clear_internal_node_names(tree):
    """
    Remove the names of all internal (non-terminal) nodes in the tree.
    """
    for node in tree.get_nonterminals():
        node.name = None
    return tree

def main(input_file, output_tree, PARSIMONY_ALGORITHM = "Camin-Sokal"):
    taxa, matrix = parse_input_matrix(input_file)
    tree = build_tree(taxa, matrix)

    if PARSIMONY_ALGORITHM == "Camin-Sokal":
        score, _ = camin_sokal_parsimony(tree, taxa, matrix)
    elif PARSIMONY_ALGORITHM == "Dollo":
        score, _ = dollo_parsimony(tree, taxa, matrix)
    elif PARSIMONY_ALGORITHM == "Wagner":
        score, _ = wagner_parsimony(tree, taxa, matrix)
    elif PARSIMONY_ALGORITHM == "Fitch":
        score, _ = fitch_parsimony(tree, taxa, matrix)
    else:
        raise ValueError(f"Unsupported algorithm: {PARSIMONY_ALGORITHM}")

    print("Constructed Phylogenetic Tree:")
    Phylo.draw_ascii(tree)
    print(f"\n{PARSIMONY_ALGORITHM} parsimony score: {score}")
    
    # Clear intermediate node names
    tree = clear_internal_node_names(tree)

    with open(output_tree, "w") as tree_file:
        Phylo.write(tree, tree_file, "newick")

    # Return name of the created tree file
    return tree_file
