from argparse import ArgumentParser
from pathlib import Path

from processing.ast_parsers.tree_sitter_parsers import *

parser = ArgumentParser()
parser.add_argument("--input", type=str, required=True, help="Path to the folder with Python train/validation/test")
parser.add_argument("--output", type=str, required=True, help="Path to the folder to store parsed trees")
parser.add_argument("--language", type=str, required=True, help="Select language for parsing")
args = parser.parse_args()

lang = args.language.lower()

if lang == "go":
    parser = GoAstParser()
elif lang == "java":
    parser = JavaAstParser()
elif lang == "js" or lang == "javascript":
    parser = JavascriptAstParser()
elif lang == "php":
    parser = PhpAstParser()
elif lang == "py" or lang == "python":
    parser = PythonAstParser()
elif lang == "ruby":
    parser = RubyAstParser()
else:
    raise ValueError(f"Invalid language {lang}")

parser.parse_subfolders(Path(args.input), Path(args.output))
