"""Install the adapter from a CLI model path or one inherited by a worker."""
import importlib.abc
import importlib.machinery
import logging
import os
import sys

logger = logging.getLogger(__name__)


def _model_path_from_args(args):
    model_path = None
    arguments = iter(args)
    for argument in arguments:
        if argument == '--':
            break
        if argument == '--model-path':
            value = next(arguments, None)
            model_path = value if value and not value.startswith('-') else None
        elif argument.startswith('--model-path='):
            model_path = argument.partition('=')[2] or None
    return model_path


class EngramLoader(importlib.abc.Loader):
    def __init__(self, original):
        self.original = original

    def create_module(self, spec):
        return self.original.create_module(spec)

    def exec_module(self, module):
        self.original.exec_module(module)
        if module.__name__ == 'sglang.srt.layers.engram':
            from engram_backend import install
            install(module)
        elif module.__name__ == 'sglang.srt.layers.attention.dsv4.metadata':
            # V4.1 ratio-1/2 indexers always call the FP4 DeepGEMM kernel.
            # SM120 needs its split-128 planner even when the legacy FP8
            # indexer uses the torch path. The upstream guard misses this case.
            cls = module.PagedIndexerMetadata
            original = cls.__post_init__
            def post_init(self):
                if module._IS_SM120 and self.compress_ratio in (1, 2):
                    self.force_deep_gemm_metadata = True
                original(self)
            cls.__post_init__ = post_init
        elif module.__name__ == 'sglang.srt.layers.attention.dsv4.v41_indexer':
            _patch_candidate_indexer(module)


def _patch_candidate_indexer(module):
    # sgl-deep-gemm ships no SM120 paged sparse MQA logits kernel (the sparse
    # candidate table is SM100-only) and upstream make_candidate_indexer
    # raises in that case. Fall back to the dense candidate blocks path (the
    # SM90 route), which only uses kernels SM120 already has.
    original = module.make_candidate_indexer

    def make_candidate_indexer(*args, **kwargs):
        from sglang.srt.layers.deep_gemm_wrapper.configurer import (
            DEEPGEMM_PAGED_SPARSE_MQA_LOGITS,
        )
        if (
            kwargs.get('candidate_topk_blocks', 0) > 0
            and module.is_sm100_or_newer()
            and not DEEPGEMM_PAGED_SPARSE_MQA_LOGITS
        ):
            logger.warning(
                'DeepGEMM paged sparse MQA logits unavailable on this GPU; '
                'using the dense candidate blocks indexer'
            )
            from sglang.srt.layers.attention.dsv4.v41_indexer.dense_blocks import (
                DenseBlocksBackend,
            )
            dense = DenseBlocksBackend(
                token_to_kv_pool=kwargs['token_to_kv_pool'],
                req_to_token=kwargs['req_to_token'],
                candidate_topk_blocks=kwargs['candidate_topk_blocks'],
                candidate_block_size=kwargs['candidate_block_size'],
                use_deep_gemm_prefill=module._use_deep_gemm_prefill(),
            )
            return dense, dense
        return original(*args, **kwargs)

    module.make_candidate_indexer = make_candidate_indexer


class EngramFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname not in ('sglang.srt.layers.engram',
                            'sglang.srt.layers.attention.dsv4.metadata',
                            'sglang.srt.layers.attention.dsv4.v41_indexer'):
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is not None:
            spec.loader = EngramLoader(spec.loader)
        return spec

# Python loads this before the launcher, while its CLI arguments are available.
# Spawned workers may start without those arguments, so pass the resolved path
# through the environment. The CLI takes precedence over a legacy explicit value.
_model_path = _model_path_from_args(sys.argv[1:])
if _model_path:
    os.environ['DSV41_SOURCE'] = os.path.abspath(os.path.expanduser(_model_path))

if os.environ.get('DSV41_SOURCE'):
    sys.meta_path.insert(0, EngramFinder())
