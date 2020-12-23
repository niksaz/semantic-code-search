import os
import multiprocessing
import random
import time
from abc import ABC, abstractmethod
from collections import defaultdict, OrderedDict
from enum import Enum, auto
from typing import List, Dict, Any, Iterable, Tuple, Optional, Union, Callable, Type, DefaultDict
import tqdm

import numpy as np
import wandb
import tensorflow as tf
from dpu_utils.utils import RichPath

from utils import data_pipeline
from utils.py_utils import run_jobs_in_parallel
from encoders import Encoder, QueryType


class StaticModel(Model):

    @classmethod
    def get_default_hyperparameters(cls) -> Dict[str, Any]:
        hypers = {}
        model_hypers = {
            'type': 'bm25',
            'batch_size': 1000,
            'vocab_size': 10000,

            # BM-25 coefficients
            'k1': 1.2,
            'b': 0.75
        }
        hypers.update(super().get_default_hyperparameters())
        hypers.update(model_hypers)
        return hypers

    def __init__(self,
                 hyperparameters: Dict[str, Any],
                 run_name: str = None,
                 model_save_dir: Optional[str] = None,
                 log_save_dir: Optional[str] = None):
        super().__init__(
            hyperparameters,
            code_encoder_type=None,
            query_encoder_type=None,
            run_name=run_name,
            model_save_dir=model_save_dir,
            log_save_dir=log_save_dir)

        self.df = {}
        self.vocabulary = None
        self.n_docs = 0
        self.avgdl = 0

    def save(self, path: RichPath) -> None:
        weights_to_save = {'k1': self.k1, 'b': self.b}

        data_to_save = {
                         "model_type": type(self).__name__,
                         "hyperparameters": self.hyperparameters,
                         "query_metadata": self.__query_metadata,
                         "per_code_language_metadata": self.__per_code_language_metadata,
                         "weights": weights_to_save,
                         "run_name": self.__run_name,
                       }

        path.save_as_compressed_file(data_to_save)

    def make_model(self, is_train: bool):



        self.__placeholders['df'] = tf.placeholder(input=np.zeros(shape=[self.hyperparameters['vocab_size']],
                                                                 dtype=np.float32),
                                                   shape=[self.hyperparameters['vocab_size']], name='df')
        self.__placeholders['n_docs'] = tf.placeholder(tf.float32, shape=(), name='n_docs')
        self.__placeholders['avgdl'] = tf.placeholder(tf.float32, shape=(), name='avgdl')



        code_representation_size = next(iter(self.__code_encoders.values())).output_representation_size
        query_representation_size = self.__query_encoder.output_representation_size
        assert code_representation_size == query_representation_size, \
            f'Representations produced for code ({code_representation_size}) and query ({query_representation_size}) cannot differ!'

    def load_metadata(self, data_dirs: List[RichPath], max_files_per_dir: Optional[int] = None, parallelize: bool = True) -> None:
        raw_query_metadata_list = []
        raw_code_language_metadata_lists: DefaultDict[str, List] = defaultdict(list)

        def metadata_parser_fn(_, file_path: RichPath) -> Iterable[Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]]:
            raw_query_metadata = self.__query_encoder_type.init_metadata()
            per_code_language_metadata: DefaultDict[str, Dict[str, Any]] = defaultdict(self.__code_encoder_type.init_metadata)

            for data_sample in data_pipeline.combined_samples_generator(file_path):
                sample_language = data_sample['language']
                self.__code_encoder_type.load_metadata_from_sample(data_sample,
                                                                   per_code_language_metadata[sample_language],
                                                                   self.hyperparameters['code_use_subtokens'],
                                                                   self.hyperparameters['code_mark_subtoken_end'])
                self.__query_encoder_type.load_metadata_from_sample([d.lower() for d in data_sample['docstring_tokens']],
                                                                    raw_query_metadata)
            yield (raw_query_metadata, per_code_language_metadata)

        def received_result_callback(metadata_parser_result: Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]):
            (raw_query_metadata, per_code_language_metadata) = metadata_parser_result
            raw_query_metadata_list.append(raw_query_metadata)
            for (metadata_language, raw_code_language_metadata) in per_code_language_metadata.items():
                raw_code_language_metadata_lists[metadata_language].append(raw_code_language_metadata)

        def finished_callback():
            pass

        if parallelize:
            run_jobs_in_parallel(get_data_files_from_directory(data_dirs, max_files_per_dir),
                                 metadata_parser_fn,
                                 received_result_callback,
                                 finished_callback)
        else:
            for (idx, file) in enumerate(get_data_files_from_directory(data_dirs, max_files_per_dir)):
                for res in metadata_parser_fn(idx, file):
                    received_result_callback(res)

        self.__query_metadata = self.__query_encoder_type.finalise_metadata("query", self.hyperparameters, raw_query_metadata_list)
        for (language, raw_per_language_metadata) in raw_code_language_metadata_lists.items():
            self.__per_code_language_metadata[language] = \
                self.__code_encoder_type.finalise_metadata("code", self.hyperparameters, raw_per_language_metadata)


