from argparse import ArgumentParser
from pathlib import Path

from processing.graph_parser import GraphParser

parser = ArgumentParser()
parser.add_argument("--input", type=str, required=True, help="Path to the folder with train/validation/test data")
parser.add_argument("--output", type=str, required=True, help="Path to the folder to store parsed trees")
args = parser.parse_args()

parser = GraphParser()
parser.parse_subfolders(Path(args.input), Path(args.output))
