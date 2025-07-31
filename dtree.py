import os, sys
import argparse
sys.path.append("lib")
import matrix_parsimony, main

__version__ = "1.0"
date_of_creation = "27/07/2025"

def show_banner():
    print(r"""

    ██████╗     ███████╗    ██████╗     ███████╗    ███████╗
    ██╔══██╗     ╚══██╔╝    ██╔══██╗    ██╔═══╝     ██╔═══╝
    ██║  ██║        ██║     ██████╔╝    █████╗      █████╗
    ██║  ██║        ██║     ██╔══██╗    ██╔══╝      ██╔══╝
    ██████╔╝        ██║     ██║  ██║    ███████╗    ███████╗

    Decision Tree Creator
    """)

def show_help():
    help_text = f"""
    {'='*80}
    ■ DTREE (Decision Tree Creator) v{__version__} ■
    {'='*80}
    📅 Created: {date_of_creation}
    👨‍💻 Author: Oleg Reva (oleg.reva@up.ac.za)
       Centre for Bioinformatics and Computational Biology,
       University of Pretoria, South Africa
    
    🔬 Purpose:
    Automated pipeline for creation of decision making tree algorithms.

    ⚙️ Dependencies:
    ┌───────────────────┬───────────────────────┐
    │ Tool              │ Minimum Version       │
    ├───────────────────┼───────────────────────┤
    │ Python            │ 3.12.3                │
    │ BioPython         │ 1.85                  │
    │ sklearn           │ 1.7.0                 │
    │ pandas            │ 2.2.3                 │
    │ numpy             │ 2.2.3                 │
    │ matplotlib        │ 3.10.1                │
    └───────────────────┴───────────────────────┘

    🚀 Usage:
        python3 decision_tree.py [arguments]
        
    🔧 Arguments:
        REQUIRED:
        -f, --input_file            Input matrix file in FASTA or CSV formats
        -g, --label_file            Input text file with labeled end nodes 
        
        OPTIONAL:
        -i, --input_dir             Input directory (default: 'input')
        -o, --output_dir            Output directory (default: 'output') 
        -p, --project_dir           Project directory (default: '')
        -a, --algorithm             Parsimony tree inference algorithm: Camin-Sokal, Wagner, Dollo, Fitch 
                                    (default: 'Camin-Sokal')
        -l, --tree_levels           Number of levels in decision tree (default: 3)
        -m, --min_accuracy          Minimum classification accuracy per decision node (default: 0.95)
       
    📊 Output:
    - tree.nwk:         initial parsimony tree
    - dtree.nwk:        decision making tree
    - algorithm.txt:    decision making algorithm
    
    🆘 Help Options:
        -h, --help            Show this help
        -v, --version         Show version info
    {'='*80}
    """
    print(help_text)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Decision tree Interface",
        add_help=False  # Disable default help to allow full control
    )

    # Required arguments
    parser.add_argument("-f", "--input_file", default="", help="Input file in FASTA or CSV format")
    parser.add_argument("-g", "--label_file", default="", help="File with group labels")

    # Optional arguments
    parser.add_argument("-i", "--input_dir", default="input", help="Input folder")
    parser.add_argument("-o", "--output_dir", default="output", help="Output folder")
    parser.add_argument("-p", "--project_dir", default="", help="Project folder name")
    parser.add_argument("-a", "--algorithm", choices=["Camin-Sokal", "Wagner", "Dollo", "Fitch"],
                        default="Camin-Sokal", help="Parsimony algorithm to use (default: Camin-Sokal)")
    parser.add_argument("-l", "--tree_levels", type=int, default=3, help="Number of levels in decision tree (default: 3)")
    parser.add_argument("-m", "--min_accuracy", type=float, default=0.95, help="Minimum classification accuracy (default: 0.95)")

    # Custom help and version
    parser.add_argument("-h", "--help", action="store_true", help="Show help message and exit")
    parser.add_argument("-v", "--version", action="store_true", help="Show version info and exit")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()

    if args.help:
        show_help()
        sys.exit(0)

    if args.version:
        print(f"\nVersion {__version__} created on {date_of_creation}\n")
        sys.exit(0)
        
    input_path = os.path.join(args.input_dir, args.project_dir)

    if not os.path.exists(input_path):
        print(f"\nInput path {input_path} does not exist!")
        sys.exit(1)

    input_file = os.path.join(input_path, args.input_file)
    if not os.path.exists(input_file):
        print(f"\nInput file {input_file} does not exist!")
        sys.exit(1)

    label_file = os.path.join(input_path, args.label_file)
    if not os.path.exists(label_file):
        print(f"\nLabel file {label_file} does not exist!")
        sys.exit(1)

    # Use already parsed values
    levels = args.tree_levels
    min_acc = args.min_accuracy

    # Check and create output folders
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, args.project_dir)
    os.makedirs(output_path, exist_ok=True)

    output_tree = os.path.join(
        output_path,
        os.path.splitext(os.path.basename(args.input_file))[0] + ".tre"
    )

    # Generate tree file (does not need to return anything)
    matrix_parsimony.main(input_file, output_tree, args.algorithm)

    # Run decision tree classifier
    main.main(output_tree, input_file, label_file, output_dir=output_path, levels=levels, min_acc=min_acc)
