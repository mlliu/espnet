import argparse
import logging
from typing import Callable, Collection, Dict, List, Optional, Tuple

import numpy as np
import torch
from typeguard import check_argument_types, check_return_type

from espnet2.asr.ctc import CTC
from espnet2.asr.decoder.abs_decoder import AbsDecoder
from espnet2.asr.decoder.rnn_decoder import RNNDecoder
from espnet2.asr.decoder.transformer_decoder import (
    DynamicConvolution2DTransformerDecoder,
    DynamicConvolutionTransformerDecoder,
    LightweightConvolution2DTransformerDecoder,
    LightweightConvolutionTransformerDecoder,
    TransformerDecoder,
)
# from espnet2.asr.discrete_asr_espnet_model import ESPnetDiscreteASRModel
from espnet2.asr.discrete_asr_espnet_model_multi_input import ESPnetDiscreteASRMultiInputModel
from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.asr.encoder.branchformer_encoder import BranchformerEncoder
from espnet2.asr.encoder.conformer_encoder import ConformerEncoder
from espnet2.asr.encoder.contextual_block_transformer_encoder import (
    ContextualBlockTransformerEncoder,
)
from espnet2.asr.encoder.e_branchformer_encoder import EBranchformerEncoder
from espnet2.asr.encoder.rnn_encoder import RNNEncoder
from espnet2.asr.encoder.transformer_encoder import TransformerEncoder
from espnet2.asr.encoder.vgg_rnn_encoder import VGGRNNEncoder
from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.asr.postencoder.abs_postencoder import AbsPostEncoder
from espnet2.asr.postencoder.hugging_face_transformers_postencoder import (
    HuggingFaceTransformersPostEncoder,
)
from espnet2.asr.preencoder.abs_preencoder import AbsPreEncoder
from espnet2.asr.preencoder.linear import LinearProjection
from espnet2.asr.preencoder.sinc import LightweightSincConvs
from espnet2.asr.specaug.abs_specaug import AbsSpecAug
from espnet2.asr.specaug.specaug import SpecAug
from espnet2.mt.espnet_model import ESPnetMTModel
from espnet2.mt.frontend.embedding import Embedding,Embedding_multi_input
from espnet2.tasks.abs_task import AbsTask
from espnet2.text.phoneme_tokenizer import g2p_choices
from espnet2.torch_utils.initialize import initialize
from espnet2.train.abs_espnet_model import AbsESPnetModel
from espnet2.train.class_choices import ClassChoices
from espnet2.train.collate_fn import CommonCollateFn
from espnet2.train.preprocessor import MutliTokenizerCommonPreprocessor
from espnet2.train.trainer import Trainer
from espnet2.utils.get_default_kwargs import get_default_kwargs
from espnet2.utils.nested_dict_action import NestedDictAction
from espnet2.utils.types import int_or_none, str2bool, str_or_none
from espnet2.utils.types import (
    humanfriendly_parse_size_or_none,
    int_or_none,
    str2bool,
    str2triple_str,
    str_or_int,
    str_or_none,
)
from espnet2.train.dataset import DATA_TYPES, AbsDataset, ESPnetDataset
from espnet.nets.pytorch_backend.nets_utils import pad_list
from typing import Collection, Dict, List, Tuple, Union

frontend_choices = ClassChoices(
    name="frontend",
    classes=dict(
        embed=Embedding,
        embed_multi_input=Embedding_multi_input,
    ),
    type_check=AbsFrontend,
    default="embed",
)
specaug_choices = ClassChoices(
    name="specaug",
    classes=dict(
        specaug=SpecAug,
    ),
    type_check=AbsSpecAug,
    default=None,
    optional=True,
)
preencoder_choices = ClassChoices(
    name="preencoder",
    classes=dict(
        sinc=LightweightSincConvs,
        linear=LinearProjection,
    ),
    type_check=AbsPreEncoder,
    default=None,
    optional=True,
)
encoder_choices = ClassChoices(
    "encoder",
    classes=dict(
        conformer=ConformerEncoder,
        transformer=TransformerEncoder,
        contextual_block_transformer=ContextualBlockTransformerEncoder,
        vgg_rnn=VGGRNNEncoder,
        rnn=RNNEncoder,
        branchformer=BranchformerEncoder,
        e_branchformer=EBranchformerEncoder,
    ),
    type_check=AbsEncoder,
    default="rnn",
)
postencoder_choices = ClassChoices(
    name="postencoder",
    classes=dict(
        hugging_face_transformers=HuggingFaceTransformersPostEncoder,
    ),
    type_check=AbsPostEncoder,
    default=None,
    optional=True,
)
decoder_choices = ClassChoices(
    "decoder",
    classes=dict(
        transformer=TransformerDecoder,
        lightweight_conv=LightweightConvolutionTransformerDecoder,
        lightweight_conv2d=LightweightConvolution2DTransformerDecoder,
        dynamic_conv=DynamicConvolutionTransformerDecoder,
        dynamic_conv2d=DynamicConvolution2DTransformerDecoder,
        rnn=RNNDecoder,
    ),
    type_check=AbsDecoder,
    default="rnn",
)
model_choices = ClassChoices(
    "model",
    classes=dict(
        mt=ESPnetMTModel,
        discrete_asr_multi_input=ESPnetDiscreteASRMultiInputModel,
    ),
    type_check=AbsESPnetModel,
    default="mt",
)


class MTTask_multi_input(AbsTask):
    # If you need more than one optimizers, change this value
    num_optimizers: int = 1

    # Add variable objects configurations
    class_choices_list = [
        # --frontend and --frontend_conf
        frontend_choices,
        # --specaug and --specaug_conf
        specaug_choices,
        # --preencoder and --preencoder_conf
        preencoder_choices,
        # --encoder and --encoder_conf
        encoder_choices,
        # --postencoder and --postencoder_conf
        postencoder_choices,
        # --decoder and --decoder_conf
        decoder_choices,
        # --model and --model_conf
        model_choices,
    ]

    # If you need to modify train() or eval() procedures, change Trainer class here
    trainer = Trainer

    @classmethod
    def add_task_arguments(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group(description="Task related")

        # NOTE(kamo): add_arguments(..., required=True) can't be used
        # to provide --print_config mode. Instead of it, do as
        required = parser.get_default("required")
        required += ["src_token_list_1", "token_list"]

        group.add_argument(
            "--token_list",
            type=str_or_none,
            default=None,
            help="A text mapping int-id to token (for target language)",
        )
        group.add_argument(
            "--src_token_list_1",
            type=str_or_none,
            default=None,
            help="A text mapping int-id to token (for source language)",
        )
        group.add_argument(
            "--src_token_list_2",
            type=str_or_none,
            default=None,
            help="A text mapping int-id to token (for source language)",
        )
        group.add_argument(
            "--src_token_list_3",
            type=str_or_none,
            default=None,
            help="A text mapping int-id to token (for source language)",
        )
        group.add_argument(
            "--src_token_list_4",
            type=str_or_none,
            default=None,
            help="A text mapping int-id to token (for source language)",
        )
        group.add_argument(
            "--src_token_type_1",
            type=str_or_none,
            default=None,
            choices=["bpe", "char", "word", "phn", None,],
            help="The source text will be tokenized " "in the specified level token",
        )
        group.add_argument(
            "--src_token_type_2",
            type=str_or_none,
            default=None,
            choices=["bpe", "char", "word", "phn", None,],
            help="The source text will be tokenized " "in the specified level token",
        )
        group.add_argument(
            "--src_token_type_3",
            type=str_or_none,
            default=None,
            choices=["bpe", "char", "word", "phn", None,],
            help="The source text will be tokenized " "in the specified level token",
        )
        group.add_argument(
            "--src_token_type_4",
            type=str_or_none,
            default=None,
            choices=["bpe", "char", "word", "phn", None,],
            help="The source text will be tokenized " "in the specified level token",
        )

        group.add_argument(
            "--init",
            type=lambda x: str_or_none(x.lower()),
            default=None,
            help="The initialization method",
            choices=[
                "chainer",
                "xavier_uniform",
                "xavier_normal",
                "kaiming_uniform",
                "kaiming_normal",
                None,
            ],
        )

        group.add_argument(
            "--input_size",
            type=int_or_none,
            default=None,
            help="The number of input dimension of the feature",
        )
        group.add_argument(
            "--ctc_conf",
            action=NestedDictAction,
            default=get_default_kwargs(CTC),
            help="The keyword arguments for CTC class.",
        )

        group = parser.add_argument_group(description="Preprocess related")
        group.add_argument(
            "--use_preprocessor",
            type=str2bool,
            default=True,
            help="Apply preprocessing to data or not",
        )
        group.add_argument(
            "--token_type",
            type=str,
            default="bpe",
            choices=["bpe", "char", "word", "phn"],
            help="The target text will be tokenized " "in the specified level token",
        )
        group.add_argument(
            "--:q",
            type=str,
            default="bpe",
            choices=["bpe", "char", "word", "phn"],
            help="The source text will be tokenized " "in the specified level token",
        )
        group.add_argument(
            "--bpemodel",
            type=str_or_none,
            default=None,
            help="The model file of sentencepiece (for target language)",
        )
        group.add_argument(
            "--src_bpemodel_1",
            type=str_or_none,
            default=None,
            help="The model file of sentencepiece (for source language)",
        )
        group.add_argument(
            "--src_bpemodel_2",
            type=str_or_none,
            default=None,
            help="The model file of sentencepiece (for source language)",
        )
        group.add_argument(
            "--src_bpemodel_3",
            type=str_or_none,
            default=None,
            help="The model file of sentencepiece (for source language)",
        )
        group.add_argument(
            "--src_bpemodel_4",
            type=str_or_none,
            default=None,
            help="The model file of sentencepiece (for source language)",
        )
        parser.add_argument(
            "--non_linguistic_symbols",
            type=str_or_none,
            help="non_linguistic_symbols file path",
        )
        parser.add_argument(
            "--cleaner",
            type=str_or_none,
            choices=[None, "tacotron", "jaconv", "vietnamese"],
            default=None,
            help="Apply text cleaning",
        )
        parser.add_argument(
            "--g2p",
            type=str_or_none,
            choices=g2p_choices,
            default=None,
            help="Specify g2p method if --token_type=phn",
        )
        parser.add_argument(
            "--tokenizer_encode_conf",
            type=dict,
            default=None,
            help="Tokenization encoder conf, "
            "e.g. BPE dropout: enable_sampling=True, alpha=0.1, nbest_size=-1",
        )
        parser.add_argument(
            "--src_tokenizer_encode_conf",
            type=dict,
            default=None,
            help="Src tokenization encoder conf, "
            "e.g. BPE dropout: enable_sampling=True, alpha=0.1, nbest_size=-1",
        )

        for class_choices in cls.class_choices_list:
            # Append --<name> and --<name>_conf.
            # e.g. --encoder and --encoder_conf
            class_choices.add_arguments(group)

    @classmethod
    def build_collate_fn(
        cls, args: argparse.Namespace, train: bool
    ) -> Callable[
        [Collection[Tuple[str, Dict[str, np.ndarray]]]],
        Tuple[List[str], Dict[str, torch.Tensor]],
    ]:
        assert check_argument_types()
        # NOTE(kamo): int value = 0 is reserved by CTC-blank symbol
        #return CommonCollateFn(float_pad_value=0.0, int_pad_value=-1)
        return CollateFnMultiInput(float_pad_value=0.0, int_pad_value=-1)





    @classmethod
    def build_preprocess_fn(
        cls, args: argparse.Namespace, train: bool
    ) -> Optional[Callable[[str, Dict[str, np.array]], Dict[str, np.ndarray]]]:
        assert check_argument_types()
        if args.use_preprocessor:
            retval = MutliTokenizerCommonPreprocessor(
                train=train,
                token_type=[args.token_type, args.src_token_type_1,args.src_token_type_2,args.src_token_type_3,args.src_token_type_4],
                token_list=[args.token_list, args.src_token_list_1,args.src_token_list_2,args.src_token_list_3,args.src_token_list_4],
                bpemodel=[args.bpemodel, args.src_bpemodel_1,args.src_bpemodel_2,args.src_bpemodel_3,args.src_bpemodel_4],
                non_linguistic_symbols=args.non_linguistic_symbols,
                text_cleaner=args.cleaner,
                g2p_type=args.g2p,
                text_name=["text", "src_text_1", "src_text_2", "src_text_3", "src_text_4"],
                tokenizer_encode_conf=[
                    args.tokenizer_encode_conf,
                    args.src_tokenizer_encode_conf,
                ]
                if train
                else [dict(), dict()],
            )
        else:
            retval = None
        assert check_return_type(retval)
        return retval

    @classmethod
    def required_data_names(
        cls, train: bool = True, inference: bool = False
    ) -> Tuple[str, ...]:
        if not inference:
            # retval = ("src_text", "text") # because we have build_preprocess_fn
            retval = ("src_text_1", "text")


        else:
            # Recognition mode
            #retval = ("src_text",)
            #retval = ("src_text_1", "src_text_2")

            retval = ("src_text_1",)
        return retval

    @classmethod
    def optional_data_names(
        cls, train: bool = True, inference: bool = False
    ) -> Tuple[str, ...]:
        if not inference:
            retval = ("src_text_2", "src_text_3", "src_text_4")
        else:
            retval = ("src_text_2", "src_text_3", "src_text_4")
        assert check_return_type(retval)
        return retval

    @classmethod
    def build_model(cls, args: argparse.Namespace) -> ESPnetMTModel:
        assert check_argument_types()
        if isinstance(args.token_list, str):
            with open(args.token_list, encoding="utf-8") as f:
                token_list = [line.rstrip() for line in f]

            # Overwriting token_list to keep it as "portable".
            args.token_list = list(token_list)
        elif isinstance(args.token_list, (tuple, list)):
            token_list = list(args.token_list)
        else:
            raise RuntimeError("token_list must be str or list")
        vocab_size = len(token_list)
        logging.info(f"Vocabulary size: {vocab_size }")

        # if args.src_token_list is not None:
        #     if isinstance(args.src_token_list, str):
        #         with open(args.src_token_list, encoding="utf-8") as f:
        #             src_token_list = [line.rstrip() for line in f]
        #
        #         # Overwriting src_token_list to keep it as "portable".
        #         args.src_token_list = list(src_token_list)
        #     elif isinstance(args.src_token_list, (tuple, list)):
        #         src_token_list = list(args.src_token_list)
        #     else:
        #         raise RuntimeError("token_list must be str or list")
        #     src_vocab_size = len(src_token_list)
        #     logging.info(f"Source vocabulary size: {src_vocab_size }")
        # else:
        #     src_token_list, src_vocab_size = None, None

        # process source token list 1
        if args.src_token_list_1 is not None:
            if isinstance(args.src_token_list_1, str):
                with open(args.src_token_list_1, encoding="utf-8") as f:
                    src_token_list_1 = [line.rstrip() for line in f]

                # Overwriting src_token_list to keep it as "portable".
                args.src_token_list_1 = list(src_token_list_1) # the type of src_token_list_1 is list
            elif isinstance(args.src_token_list_1, (tuple, list)):
                src_token_list_1 = list(args.src_token_list_1)
            else:
                raise RuntimeError("token_list must be str or list")
            src_vocab_size_1 = len(src_token_list_1)
            logging.info(f"Source vocabulary size: {src_vocab_size_1 }")
        else:
            src_token_list_1, src_vocab_size_1 = None, None

        # process source token list 2
        if args.src_token_list_2 is not None:
            if isinstance(args.src_token_list_2, str):
                with open(args.src_token_list_2, encoding="utf-8") as f:
                    src_token_list_2 = [line.rstrip() for line in f]

                # Overwriting src_token_list to keep it as "portable".
                args.src_token_list_2 = list(src_token_list_2)
            elif isinstance(args.src_token_list_2, (tuple, list)):
                src_token_list_2 = list(args.src_token_list_2)
            else:
                raise RuntimeError("token_list must be str or list")
            src_vocab_size_2 = len(src_token_list_2)
            logging.info(f"Source vocabulary size: {src_vocab_size_2 }")
        else:
            src_token_list_2, src_vocab_size_2 = None, None

        # process source token list 3
        if args.src_token_list_3 is not None:
            if isinstance(args.src_token_list_3, str):
                with open(args.src_token_list_3, encoding="utf-8") as f:
                    src_token_list_3 = [line.rstrip() for line in f]

                # Overwriting src_token_list to keep it as "portable".
                args.src_token_list_3 = list(src_token_list_3)
            elif isinstance(args.src_token_list_3, (tuple, list)):
                src_token_list_3 = list(args.src_token_list_3)
            else:
                raise RuntimeError("token_list must be str or list")
            src_vocab_size_3 = len(src_token_list_3)
            logging.info(f"Source vocabulary size: {src_vocab_size_3 }")
        else:
            src_token_list_3, src_vocab_size_3 = None, None

        # process source token list 4
        if args.src_token_list_4 is not None:
            if isinstance(args.src_token_list_4, str):
                with open(args.src_token_list_4, encoding="utf-8") as f:
                    src_token_list_4 = [line.rstrip() for line in f]

                # Overwriting src_token_list to keep it as "portable".
                args.src_token_list_4 = list(src_token_list_4)
            elif isinstance(args.src_token_list_4, (tuple, list)):
                src_token_list_4 = list(args.src_token_list_4)
            else:
                raise RuntimeError("token_list must be str or list")
            src_vocab_size_4 = len(src_token_list_4)
            logging.info(f"Source vocabulary size: {src_vocab_size_4 }")
        else:
            src_token_list_4, src_vocab_size_4 = None, None


        # 1. frontend
        if args.input_size is None:
            # Extract features in the model
            frontend_class = frontend_choices.get_class(args.frontend)
            # frontend = frontend_class(input_size=src_vocab_size, **args.frontend_conf)
            frontend = frontend_class(input_size=[src_vocab_size_1,src_vocab_size_2,src_vocab_size_3,src_vocab_size_4], **args.frontend_conf)
            input_size = frontend.output_size()
        else:
            # Give features from data-loader
            args.frontend = None
            args.frontend_conf = {}
            frontend = None
            input_size = args.input_size

        # 2. Data augmentation for spectrogram
        if getattr(args, "specaug", None) is not None:
            specaug_class = specaug_choices.get_class(args.specaug)
            specaug = specaug_class(**args.specaug_conf)
        else:
            specaug = None

        # 3. Pre-encoder input block
        # NOTE(kan-bayashi): Use getattr to keep the compatibility
        if getattr(args, "preencoder", None) is not None:
            preencoder_class = preencoder_choices.get_class(args.preencoder)
            preencoder = preencoder_class(**args.preencoder_conf)
            input_size = preencoder.output_size()
        else:
            preencoder = None

        # 4. Encoder
        encoder_class = encoder_choices.get_class(args.encoder)
        encoder = encoder_class(input_size=input_size, **args.encoder_conf)

        # 5. Post-encoder block
        # NOTE(kan-bayashi): Use getattr to keep the compatibility
        encoder_output_size = encoder.output_size()
        if getattr(args, "postencoder", None) is not None:
            postencoder_class = postencoder_choices.get_class(args.postencoder)
            postencoder = postencoder_class(
                input_size=encoder_output_size, **args.postencoder_conf
            )
            encoder_output_size = postencoder.output_size()
        else:
            postencoder = None

        # 5. Decoder
        decoder_class = decoder_choices.get_class(args.decoder)

        decoder = decoder_class(
            vocab_size=vocab_size,
            encoder_output_size=encoder_output_size,
            **args.decoder_conf,
        )

        # 6. CTC
        ctc = CTC(
            odim=vocab_size, encoder_output_size=encoder_output_size, **args.ctc_conf
        )

        # 8. Build model
        try:
            model_class = model_choices.get_class(args.model)
            if args.model == "discrete_asr" or args.model == "discrete_asr_multi_input":
                extra_model_conf = dict(ctc=ctc, specaug=specaug)
            else:
                extra_model_conf = dict()
        except AttributeError:
            model_class = model_choices.get_class("mt")
            extra_model_conf = dict()
        model = model_class(
            vocab_size=vocab_size,
            src_vocab_size=[src_vocab_size_1,src_vocab_size_2,src_vocab_size_3,src_vocab_size_4],
            frontend=frontend,
            preencoder=preencoder,
            encoder=encoder,
            postencoder=postencoder,
            decoder=decoder,
            token_list=token_list,
            src_token_list=[src_token_list_1,src_token_list_2,src_token_list_3,src_token_list_4],
            **args.model_conf,
            **extra_model_conf,
        )

        # FIXME(kamo): Should be done in model?
        # 9. Initialize
        if args.init is not None:
            initialize(model, args.init)

        assert check_return_type(model)
        # register forward hooks for print memory usage
        #for layer in model.children():
        #    layer.register_forward_hook(print_memory_usage)

        return model

# memory usage hook
def print_memory_usage(self, input, output):
    print(f'{self.__class__.__name__}:')
    print(f"Memory usage: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")

class CollateFnMultiInput:
    """Functor class of common_collate_fn()"""

    def __init__(
        self,
        float_pad_value: Union[float, int] = 0.0,
        int_pad_value: int = -32768,
        not_sequence: Collection[str] = (),
    ):
        assert check_argument_types()
        self.float_pad_value = float_pad_value
        self.int_pad_value = int_pad_value
        self.not_sequence = set(not_sequence)

    def __repr__(self):
        return (
            f"{self.__class__}(float_pad_value={self.float_pad_value}, "
            f"int_pad_value={self.float_pad_value})"
        )

    def __call__(
        self, data: Collection[Tuple[str, Dict[str, np.ndarray]]]
    ) -> Tuple[List[str], Dict[str, torch.Tensor]]:
        return collate_fn_multi_input(
            data,
            float_pad_value=self.float_pad_value,
            int_pad_value=self.int_pad_value,
            not_sequence=self.not_sequence,
        )

def collate_fn_multi_input(
    data: Collection[Tuple[str, Dict[str, np.ndarray]]],
    float_pad_value: Union[float, int] = 0.0,
    int_pad_value: int = -32768,
    not_sequence: Collection[str] = (),
) -> Tuple[List[str], Dict[str, torch.Tensor]]:
    """Concatenate ndarray-list to an array and convert to torch.Tensor.

    Examples:
        >>> from espnet2.samplers.constant_batch_sampler import ConstantBatchSampler,
        >>> import espnet2.tasks.abs_task
        >>> from espnet2.train.dataset import ESPnetDataset
        >>> sampler = ConstantBatchSampler(...)
        >>> dataset = ESPnetDataset(...)
        >>> keys = next(iter(sampler)
        >>> batch = [dataset[key] for key in keys]
        >>> batch = common_collate_fn(batch)
        >>> model(**batch)

        Note that the dict-keys of batch are propagated from
        that of the dataset as they are.

    """
    # input data is like:
    # batch = ("sample_id", {"src_text_1": tensor, "src_text_2": tensor, "src_text_3": tensor, "src_text_4": tensor, "text": tensor})
    # data = [
    #     ("sample_id", {"src_text_1": tensor, "src_text_2": tensor, "src_text_3": tensor, "src_text_4": tensor, "text": tensor}),
    #     ("sample_id2", {"src_text_1": tensor, "src_text_2": tensor, "src_text_3": tensor, "src_text_4": tensor, "text": tensor}),
    #     ...
    # ]

    # what we want to do is to concate the src_text_1, src_text_2, src_text_3, src_text_4 to src_text
    # since we know that these are all the same length, we can just concate them along the last axis
    # for the lengths, we can just take the length of the first one

    # output data is like:
    # batch = {"src_text": tensor, "src_text_lengths": tensor, "text": tensor, "text_lengths": tensor}
    assert check_argument_types()
    uttids = [u for u, _ in data] # a list of uttids
    data = [d for _, d in data] # a list of dict

    assert all(set(data[0]) == set(d) for d in data), "dict-keys mismatching"
    assert all(
        not k.endswith("_lengths") for k in data[0]
    ), f"*_lengths is reserved: {list(data[0])}"

    # assert that the keys include src_text_1, src_text_2, src_text_3, src_text_4, text
    assert "src_text_1" in data[0]
    # number of src_text
    num_src_text = 0
    for key in data[0]:
        if key.startswith("src_text"):
            num_src_text += 1

    output = {}
    # key is like 'src_text_1', 'src_text_2', ...
    for key in data[0]:
        # NOTE(kamo):
        # Each models, which accepts these values finally, are responsible
        # to repaint the pad_value to the desired value for each tasks.
        if data[0][key].dtype.kind == "i":
            pad_value = int_pad_value
        else:
            pad_value = float_pad_value

        array_list = [d[key] for d in data]

        # Assume the first axis is length:
        # tensor_list: Batch x (Length, ...)
        tensor_list = [torch.from_numpy(a) for a in array_list]
        # tensor: (Batch, Length, ...)
        tensor = pad_list(tensor_list, pad_value)
        output[key] = tensor

        # lens: (Batch,)
        if key not in not_sequence:
            lens = torch.tensor([d[key].shape[0] for d in data], dtype=torch.long)
            output[key + "_lengths"] = lens
    # print the shape of src_text_1, which should be (Batch, Length)
    # print("src_text_1 shape: ", output["src_text_1"].shape)
    # concate the src_text_1, src_text_2 if exist, src_text_3 if exist, src_text_4 if exist to src_text if there are more than one src_text

    tmp_src_text = []

    for i in range(1, num_src_text + 1):
        tmp_src_text.append(output[f"src_text_{i}"])
    # check if the dimension matched for concating
    # if not matched, we need to pad the tensor to make them have the same dimension
    # if matched, we can just concate them

    output["src_text"] = torch.stack(tmp_src_text, dim=-1)

    #output["src_text"] = torch.cat([output["src_text_1"], output["src_text_2"], output["src_text_3"], output["src_text_4"]], dim=-1) #shape: (Batch, Length, ...)
    output["src_text_lengths"] = output["src_text_1_lengths"]

    # remove the src_text_1, src_text_2, src_text_3, src_text_4
    del output["src_text_1"] # remove the src_text_1
    if "src_text_2" in output:
        del output["src_text_2"]
    if "src_text_3" in output:
        del output["src_text_3"]
    if "src_text_4" in output:
        del output["src_text_4"]
    # remove the src_text_1_lengths, src_text_2_lengths, src_text_3_lengths, src_text_4_lengths
    del output["src_text_1_lengths"] # remove the src_text_1_lengths
    if "src_text_2_lengths" in output:
        del output["src_text_2_lengths"]
    if "src_text_3_lengths" in output:
        del output["src_text_3_lengths"]
    if "src_text_4_lengths" in output:
        del output["src_text_4_lengths"]


    # print the shape of src_text, which should be (Batch, Length, 4)
    # print("src_text shape: ", output["src_text"].shape)
    # # print the shape of src_text_lengths
    # print("src_text_lengths shape: ", output["src_text_lengths"].shape)

    output = (uttids, output)
    assert check_return_type(output)
    return output
