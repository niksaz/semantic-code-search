from abc import ABC
from typing import Dict, Any

from processing.data_processor import DataProcessor


class CodeParser(DataProcessor, ABC):
    def extract_from_raw_data(self, raw_json_object: Dict[str, Any]) -> Any:
        return raw_json_object['code']
