import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any

from dpu_utils.utils import RichPath
from tqdm import tqdm


class DataProcessor(ABC):

    def __init__(self):
        self.skipped = 0
        self.failed = 0

    def parse_subfolders(self, input_folder: Path, output_folder: Path) -> None:
        self.skipped = 0
        self.failed = 0

        subfolders = os.listdir(input_folder)
        input_subfolders = [input_folder / f for f in subfolders]
        output_subfolders = [output_folder / f for f in subfolders]
        for input_subfolder, output_subfolder in zip(input_subfolders, output_subfolders):
            self.parse_folder(input_subfolder, output_subfolder)

        print(f"Skipped {self.skipped} files\nFailed on {self.failed} files")

    def parse_folder(self, input_folder: Path, output_folder: Path) -> None:
        if not output_folder.exists():
            os.makedirs(output_folder)

        files = [input_folder / file for file in os.listdir(input_folder)]
        for file in files:
            self.parse_jsonl_file(file, output_folder)

    def parse_jsonl_file(self, file: Path, output_folder: Path) -> None:
        print(f"Parsing data in {file}")
        input_file = RichPath.create(str(file))
        output_file = RichPath.create(str(output_folder / input_file.basename()))
        parsed_code = [
            self.process_data(self.extract_from_raw_data(raw_json_object))
            for raw_json_object in tqdm(input_file.read_by_file_suffix())
        ]
        print(f"Saving processed data in {output_file}")
        output_file.save_as_compressed_file(parsed_code)

    @abstractmethod
    def extract_from_raw_data(self, raw_json_object: Dict[str, Any]) -> Any:
        pass

    @abstractmethod
    def process_data(self, code: str) -> Dict[str, Any]:
        pass
