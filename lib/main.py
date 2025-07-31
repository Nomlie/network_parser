import pandas as pd
import numpy as np
from Bio import Phylo
from Bio import SeqIO
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from collections import defaultdict
import os
from datetime import datetime

# STEP 1. Compute root-to-leaf distances
def get_leaf_distances(tree):
    distances = {}
    for terminal in tree.get_terminals():
        distances[terminal.name] = tree.distance(tree.root, terminal)
    return distances

# STEP 2. Load character matrix (CSV or FASTA)
def load_character_matrix(matrix_file):
    ext = os.path.splitext(matrix_file)[1].lower()
    
    if ext == ".csv":
        df = pd.read_csv(matrix_file, index_col=0)
    elif ext in [".fa", ".fasta", ".fst", ".matrix"]:
        records = list(SeqIO.parse(matrix_file, "fasta"))
        if not records:
            raise ValueError(f"No sequences found in {matrix_file}")
        
        matrix = []
        taxa = []
        for rec in records:
            taxa.append(rec.id)
            seq = list(str(rec.seq))
            matrix.append(seq)

        lengths = {len(row) for row in matrix}
        if len(lengths) > 1:
            raise ValueError(f"Inconsistent sequence lengths in {matrix_file}: found lengths {lengths}")

        df = pd.DataFrame(matrix, index=taxa)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")
    
    return df

# STEP 3. Load label file
def load_group_labels(label_file):
    mapping = {}
    with open(label_file) as f:
        for line in f:
            if '|' in line:
                name, group = line.strip().split('|')
                mapping[name] = group
    return mapping

# STEP 4. Group leaves based on distance thresholds
def group_leaves_by_levels(distances, levels):
    leaves = list(distances.keys())
    dists = np.array([distances[leaf] for leaf in leaves])
    thresholds = np.quantile(dists, q=[i / (2 ** levels) for i in range(1, 2 ** levels)])
    
    bins = defaultdict(list)
    for leaf in leaves:
        dist = distances[leaf]
        for i, threshold in enumerate(thresholds):
            if dist <= threshold:
                bins[i].append(leaf)
                break
        else:
            bins[len(thresholds)].append(leaf)
    return bins, thresholds

# STEP 5. Build decision tree classifier for a set of strains
def build_classifier(subtree_strains, char_matrix, group_labels):
    X = char_matrix.loc[char_matrix.index.intersection(subtree_strains)]
    y = [group_labels.get(strain) for strain in X.index]

    # Remove unassigned
    filtered = [(x, label) for x, label in zip(X.index, y) if label is not None]
    if len(filtered) < 2:
        return None, None, None

    X_filtered = char_matrix.loc[[name for name, _ in filtered]]
    y_filtered = [label for _, label in filtered]

    if len(set(y_filtered)) < 2:
        return None, None, None

    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_filtered, y_filtered)
    y_pred = clf.predict(X_filtered)
    acc = accuracy_score(y_filtered, y_pred)

    return clf, acc, y_filtered

# STEP 6. Full pipeline
def main(tree_file, matrix_file, label_file, output_dir, levels=3, min_acc=0.95):
    tree = Phylo.read(tree_file, "newick")
    char_matrix = load_character_matrix(matrix_file)
    group_labels = load_group_labels(label_file)
    distances = get_leaf_distances(tree)
    bins, thresholds = group_leaves_by_levels(distances, levels)

    top_features = defaultdict(float)
    n_classified = 0
    n_total = 0

    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "decision_tree_summary.txt")
    reduced_tree_path = os.path.join(output_dir, "reduced_tree.tre")

    with open(log_path, "w") as log_file:
        log_file.write(f"Decision Tree Analysis Summary\n")
        log_file.write(f"Date: {datetime.now().isoformat()}\n")
        log_file.write(f"Levels: {levels}\nMinimum Accuracy: {min_acc}\n\n")

        log_file.write(f"=== Subtree Classifier Summary ===\n")
        print(f"\n=== Subtree Classifier Summary ===")
        for i, strains in bins.items():
            clf, acc, labels = build_classifier(strains, char_matrix, group_labels)
            if clf is None or len(strains) < 5:
                continue
            n_classified += len(labels)
            n_total += len(strains)
            msg = f" Subtree {i+1}: Accuracy = {acc:.2f} ({len(labels)} samples)\n"
            log_file.write(msg)
            print(msg.strip())
            if acc >= min_acc:
                for idx, importance in enumerate(clf.feature_importances_):
                    top_features[char_matrix.columns[idx]] += importance

        if top_features:
            log_file.write("\nTop informative characters:\n")
            print(f"\nTop informative characters for group separation:")
            for feat, score in sorted(top_features.items(), key=lambda x: x[1], reverse=True)[:10]:
                line = f" {feat}: {score:.4f}\n"
                log_file.write(line)
                print(line.strip())
        else:
            log_file.write("No subtrees achieved sufficient classification accuracy.\n")
            print("No subtrees achieved sufficient classification accuracy.")

        final_acc = n_classified / n_total if n_total > 0 else 0.0
        summary = f"\nFinal pooled accuracy: {final_acc:.3f} (on {n_classified} of {n_total} classified samples)\n"
        log_file.write(summary)
        print(summary.strip())

    for node in tree.get_nonterminals():
        node.name = None

    Phylo.write(tree, reduced_tree_path, "newick")
    print(f"Reduced tree saved to: {reduced_tree_path}")
    print(f"Decision tree summary saved to: {log_path}")
