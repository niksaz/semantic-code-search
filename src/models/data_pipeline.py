from dpu_utils.utils import RichPath

from models import ast_handler


def _original_to_raw_tree_path(file_path: RichPath, language: str):
  raw_tree_path = file_path.__str__().replace(f'/{language}/', f'/{language}_raw_trees/')
  return RichPath.create(raw_tree_path)


def combined_samples_generator(data_file: RichPath):
  raw_tree_iterator = None
  for raw_sample in data_file.read_by_file_suffix():
    if raw_tree_iterator is None:
      raw_tree_path = _original_to_raw_tree_path(data_file, language=raw_sample['language'])
      raw_tree_iterator = raw_tree_path.read_by_file_suffix()
    raw_tree = next(raw_tree_iterator)
    data_sample = ast_handler.mix_raw_tree_in(raw_sample, raw_tree)
    yield data_sample
