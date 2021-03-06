from typing import Dict, Any, Optional, Type

import tensorflow as tf
from dpu_utils.utils import RichPath

from encoders import \
    NBoWEncoder, CodeTokensASTEncoder, TBCNNEncoder, ASTNNEncoder, AstTokensEncoder, ASTPretrainedNBoWEncoder, \
    GraphPretrainedNBoWEncoder, GraphTokensEncoder, GraphNodesDataPreprocessor, \
    ASTTypeBagDataPreprocessor, TreeDataPreprocessor, TreeTokenPlusTypeDataPreprocessor
from encoders.graph_encoder import GraphEncoder
from models import Model, NeuralBoWModel, NeuralASTModel, SelfAttentionModel, ConvolutionalModel, ConvSelfAttentionModel


def get_model_class_from_name(model_name: str) -> Type[Model]:
    model_name = model_name.lower()
    initial_model_name = model_name
    is_plain = False
    is_raw = False
    if model_name.endswith('-raw'):
        is_raw = True
        model_name = model_name[:-len('-raw')]
    if model_name.endswith('-plain'):
        is_plain = True
        model_name = model_name[:-len('-plain')]

    if model_name in ['ggnn', 'ggnnmodel']:
        NeuralASTModel.MODEL_NAME = initial_model_name
        NeuralASTModel.CODE_ENCODER_TYPE = GraphTokensEncoder
        GraphEncoder.update_config(model_name, is_plain)
        return NeuralASTModel
    elif model_name in ['rnn-ggnn-sandwich']:
        NeuralASTModel.MODEL_NAME = initial_model_name
        NeuralASTModel.CODE_ENCODER_TYPE = GraphTokensEncoder
        GraphEncoder.update_config(model_name, is_plain)
        return NeuralASTModel
    elif model_name in ['transformer-ggnn-sandwich']:
        NeuralASTModel.MODEL_NAME = initial_model_name
        NeuralASTModel.CODE_ENCODER_TYPE = GraphTokensEncoder
        GraphEncoder.update_config(model_name, is_plain)
        return NeuralASTModel
    elif model_name in ['great', 'greatmodel']:
        NeuralASTModel.MODEL_NAME = initial_model_name
        NeuralASTModel.CODE_ENCODER_TYPE = GraphTokensEncoder
        GraphEncoder.update_config(model_name, is_plain)
        return NeuralASTModel
    elif model_name in ['great10', 'great10model']:
        NeuralASTModel.MODEL_NAME = initial_model_name
        NeuralASTModel.CODE_ENCODER_TYPE = GraphTokensEncoder
        GraphEncoder.update_config(model_name, is_plain)
        return NeuralASTModel
    elif model_name in ['transformer', 'transformermodel']:
        NeuralASTModel.MODEL_NAME = initial_model_name
        NeuralASTModel.CODE_ENCODER_TYPE = GraphTokensEncoder
        GraphEncoder.update_config(model_name, is_plain, is_raw)
        return NeuralASTModel
    elif model_name in ['transformer10', 'transformer10model']:
        NeuralASTModel.MODEL_NAME = initial_model_name
        NeuralASTModel.CODE_ENCODER_TYPE = GraphTokensEncoder
        GraphEncoder.update_config(model_name, is_plain, is_raw)
        return NeuralASTModel
    elif model_name in ['graphnbow', 'graphnbowmodel']:
        NeuralASTModel.MODEL_NAME = initial_model_name
        NeuralASTModel.CODE_ENCODER_TYPE = GraphTokensEncoder
        GraphEncoder.update_config(model_name, False, is_raw)
        return NeuralASTModel
    elif model_name == 'nbowtypesast':
        NeuralASTModel.MODEL_NAME = initial_model_name
        CodeTokensASTEncoder.AST_ENCODER_CLASS = NBoWEncoder
        CodeTokensASTEncoder.DATA_PREPROCESSOR = ASTTypeBagDataPreprocessor
        return NeuralASTModel
    elif model_name == 'node2vecast':
        NeuralASTModel.MODEL_NAME = initial_model_name
        CodeTokensASTEncoder.AST_ENCODER_CLASS = ASTPretrainedNBoWEncoder
        CodeTokensASTEncoder.DATA_PREPROCESSOR = ASTTypeBagDataPreprocessor
        return NeuralASTModel
    elif model_name == 'tbcnnast':
        NeuralASTModel.MODEL_NAME = initial_model_name
        CodeTokensASTEncoder.AST_ENCODER_CLASS = TBCNNEncoder
        CodeTokensASTEncoder.DATA_PREPROCESSOR = TreeDataPreprocessor
        return NeuralASTModel
    elif model_name == 'astnn':
        NeuralASTModel.MODEL_NAME = initial_model_name
        CodeTokensASTEncoder.AST_ENCODER_CLASS = ASTNNEncoder
        CodeTokensASTEncoder.CODE_ENCODER_CLASS = AstTokensEncoder
        CodeTokensASTEncoder.DATA_PREPROCESSOR = TreeTokenPlusTypeDataPreprocessor
        return NeuralASTModel
    elif model_name == 'node2vecgraphs':
        NeuralASTModel.MODEL_NAME = initial_model_name
        CodeTokensASTEncoder.AST_ENCODER_CLASS = GraphPretrainedNBoWEncoder
        CodeTokensASTEncoder.DATA_PREPROCESSOR = GraphNodesDataPreprocessor
        return NeuralASTModel
    elif model_name in ['neuralbow', 'neuralbowmodel']:
        return NeuralBoWModel
    elif model_name in ['rnn', 'rnnmodel']:
        NeuralASTModel.MODEL_NAME = initial_model_name
        NeuralASTModel.CODE_ENCODER_TYPE = GraphTokensEncoder
        GraphEncoder.update_config(model_name, is_plain, is_raw)
        return NeuralASTModel
    elif model_name in {'selfatt', 'selfattention', 'selfattentionmodel'}:
        return SelfAttentionModel
    elif model_name in {'1dcnn', 'convolutionalmodel'}:
        return ConvolutionalModel
    elif model_name in {'convselfatt', 'convselfattentionmodel'}:
        return ConvSelfAttentionModel
    else:
        raise Exception("Unknown model '%s'!" % model_name)


def restore(path: RichPath, is_train: bool, hyper_overrides: Optional[Dict[str, Any]] = None) -> Model:
    saved_data = path.read_as_pickle()

    if hyper_overrides is not None:
        saved_data['hyperparameters'].update(hyper_overrides)

    model_class = get_model_class_from_name(saved_data['model_type'])
    model = model_class(saved_data['hyperparameters'], saved_data.get('run_name'))
    model.query_metadata.update(saved_data['query_metadata'])
    for (language, language_metadata) in saved_data['per_code_language_metadata'].items():
        model.per_code_language_metadata[language] = language_metadata
    model.make_model(is_train=is_train)

    variables_to_initialize = []
    with model.sess.graph.as_default():
        with tf.name_scope("restore"):
            restore_ops = []
            used_vars = set()
            for variable in sorted(model.sess.graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES),
                                   key=lambda v: v.name):
                used_vars.add(variable.name)
                if variable.name in saved_data['weights']:
                    # print('Initializing %s from saved value.' % variable.name)
                    restore_ops.append(variable.assign(saved_data['weights'][variable.name]))
                else:
                    print('Freshly initializing %s since no saved value was found.' % variable.name)
                    variables_to_initialize.append(variable)
            for var_name in sorted(saved_data['weights']):
                if var_name not in used_vars:
                    if var_name.endswith('Adam:0') or var_name.endswith('Adam_1:0') or var_name in ['beta1_power:0',
                                                                                                    'beta2_power:0']:
                        continue
                    print('Saved weights for %s not used by model.' % var_name)
            restore_ops.append(tf.variables_initializer(variables_to_initialize))
            model.sess.run(restore_ops)
    return model
