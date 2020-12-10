import sys
import os

from lib2to3 import refactor
from typing import Dict, Any

sys.path.append(os.path.join("processing", "typilus", "src", "data_preparation", "scripts"))

from processing.code_parser import CodeParser
from graph_generator.graphgenerator import AstGraphGenerator
from graph_generator.type_lattice_generator import TypeLatticeGenerator


class GraphParser(CodeParser):

    def __init__(self):
        super().__init__()
        fixes = set(refactor.get_fixers_from_package("lib2to3.fixes"))
        self.refactorer_2to3 = refactor.RefactoringTool(fixes)
        self.lattice = TypeLatticeGenerator(
            os.path.join("processing", "typilus", "src", "data_preparation", "metadata", "typingRules.json")
        )

    def process_data(self, code: str) -> Dict[str, Any]:
        try:
            generator = AstGraphGenerator(code, self.lattice)
            graph = generator.build()
            return graph
        except (SyntaxError, UnicodeDecodeError) as e_initial:
            self.skipped += 1
            # try:
            #     refactored_code = str(self.refactorer_2to3.refactor_string(code + '\n', 'python2_code'))
            #     generator = AstGraphGenerator(refactored_code, self.lattice)
            #     graph = generator.build()
            #     return graph
            # except (SyntaxError, UnicodeDecodeError) as e:
            #     self.skipped += 1
            # except Exception as e:
            #     self.skipped += 1
        except Exception as e:
            self.failed += 1
        return {}
