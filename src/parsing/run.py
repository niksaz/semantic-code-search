from argparse import ArgumentParser
from pathlib import Path

from parsing.python_ast_parser import PythonAstParser

parser = ArgumentParser()
parser.add_argument("--input", type=str, required=True, help="Path to the folder with Python train/validation/test")
parser.add_argument("--output", type=str, required=True, help="Path to the folder to store parsed trees")
args = parser.parse_args()

parser = PythonAstParser()
parser.parse_subfolders(Path(args.input), Path(args.output))
