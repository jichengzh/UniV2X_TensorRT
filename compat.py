"""
Compatibility shim: redirects mmcv 1.x / mmdet 2.x / mmdet3d 0.x APIs
to their mmcv 2.x / mmdet 3.x / mmdet3d 1.x equivalents.

Import this module BEFORE any other UniV2X imports:
    import compat  # must be first in tools/test.py, tools/train.py etc.
"""

import sys
import types


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_module(name, **attrs):
    """Create a fake module with given attributes and install into sys.modules."""
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod
    return mod


class _ForceRegistry:
    """Wraps an mmengine Registry so that register_module always uses force=True.
    This prevents 'already registered' KeyErrors when UniV2X re-registers classes
    that happen to exist in mmdet/mmdet3d built-in registries (e.g. DiceLoss).
    """
    def __init__(self, registry):
        self._registry = registry

    def register_module(self, name=None, force=False, module=None):
        # Always force to avoid conflicts with pre-registered mmdet/mmdet3d classes
        return self._registry.register_module(name=name, force=True, module=module)

    def build(self, cfg, **kwargs):
        return self._registry.build(cfg, **kwargs)

    def get(self, key):
        return self._registry.get(key)

    def __contains__(self, item):
        return item in self._registry

    def __getattr__(self, name):
        return getattr(self._registry, name)


def _no_op_decorator(fn=None, **kwargs):
    """No-op decorator factory.  Replaces auto_fp16 / force_fp32.
    AMP is now handled at the runner/optimizer level in mmengine."""
    if fn is not None:
        return fn
    def decorator(func):
        return func
    return decorator


# ---------------------------------------------------------------------------
# 1. mmcv.runner  →  mmengine equivalents
# ---------------------------------------------------------------------------

from mmengine.model import BaseModule                        # noqa: E402
from mmengine.runner import load_checkpoint                  # noqa: E402
from mmengine.dist import get_dist_info, init_dist           # noqa: E402
from mmengine.runner.utils import set_random_seed            # noqa: E402


def wrap_fp16_model(model):
    """Convert model weights to fp16 (used before inference)."""
    return model.half()


class _EpochBasedRunnerStub:
    """Stub – only needed for import; not used at inference time."""
    pass


class _EvalHookStub:
    """Stub for mmcv.runner EvalHook (removed in mmcv 2.x)."""
    def __init__(self, *args, **kwargs): pass

class _DistEvalHookStub(_EvalHookStub):
    """Stub for mmcv.runner DistEvalHook (removed in mmcv 2.x)."""
    pass

_mmcv_runner = _make_module(
    'mmcv.runner',
    auto_fp16=_no_op_decorator,
    force_fp32=_no_op_decorator,
    BaseModule=BaseModule,
    load_checkpoint=load_checkpoint,
    get_dist_info=get_dist_info,
    init_dist=init_dist,
    wrap_fp16_model=wrap_fp16_model,
    set_random_seed=set_random_seed,
    EpochBasedRunner=_EpochBasedRunnerStub,
    EvalHook=_EvalHookStub,
    DistEvalHook=_DistEvalHookStub,
)

# mmcv.runner.fp16_utils  (direct sub-module path used in some project files)
_mmcv_runner_fp16_utils = _make_module(
    'mmcv.runner.fp16_utils',
    auto_fp16=_no_op_decorator,
    force_fp32=_no_op_decorator,
)
_mmcv_runner.fp16_utils = _mmcv_runner_fp16_utils

# mmcv.runner.base_module  (BaseModule, ModuleList, Sequential)
from mmengine.model import ModuleList as _ModuleList, Sequential as _Sequential  # noqa: E402
_mmcv_runner_base_module = _make_module(
    'mmcv.runner.base_module',
    BaseModule=BaseModule,
    ModuleList=_ModuleList,
    Sequential=_Sequential,
)
_mmcv_runner.base_module = _mmcv_runner_base_module

# mmcv.runner.hooks.hook  (HOOKS registry + Hook base class)
from mmengine.hooks import Hook as _Hook                     # noqa: E402
from mmengine.registry import MODELS as _RUNNER_MODELS       # noqa: E402 (early import)
_mmcv_runner_hooks_hook = _make_module(
    'mmcv.runner.hooks.hook',
    HOOKS=_RUNNER_MODELS,
    Hook=_Hook,
)
_mmcv_runner_hooks = _make_module(
    'mmcv.runner.hooks',
    HOOKS=_RUNNER_MODELS,
    Hook=_Hook,
    hook=_mmcv_runner_hooks_hook,
)
sys.modules['mmcv.runner.hooks.hook'] = _mmcv_runner_hooks_hook
_mmcv_runner.hooks = _mmcv_runner_hooks

# mmcv.runner.optimizer.builder  (OPTIMIZERS registry)
from mmengine.registry import OPTIMIZERS as _MMENGINE_OPTIMIZERS  # noqa: E402
_mmcv_runner_optimizer_builder = _make_module(
    'mmcv.runner.optimizer.builder',
    OPTIMIZERS=_MMENGINE_OPTIMIZERS,
)
_mmcv_runner_optimizer = _make_module(
    'mmcv.runner.optimizer',
    OPTIMIZERS=_MMENGINE_OPTIMIZERS,
    builder=_mmcv_runner_optimizer_builder,
)
sys.modules['mmcv.runner.optimizer.builder'] = _mmcv_runner_optimizer_builder
_mmcv_runner.optimizer = _mmcv_runner_optimizer


# ---------------------------------------------------------------------------
# 2. mmcv.parallel  →  mmengine + torch.nn equivalents
# ---------------------------------------------------------------------------

import torch.nn as _nn                                      # noqa: E402
from mmengine.model import MMDistributedDataParallel as _MMDDPBase  # noqa: E402


class MMDataParallel(_nn.DataParallel):
    """Drop-in replacement for mmcv 1.x MMDataParallel."""
    pass


def _scatter_data(data, device):
    """Recursively unwrap DataContainers for single-GPU inference.

    Mimics mmcv 1.x scatter logic:
    - DataContainer → DC.data[0]  (extract first GPU's group; move to device)
    - dict          → recurse per key
    - list/tuple    → recurse per element
    - Tensor        → move to device
    - everything else → pass through unchanged
    """
    # Import lazily to avoid circular import at module load time
    import torch as _t
    from collections.abc import Mapping as _Map, Sequence as _Seq

    # DataContainer will be defined below, but at call time it exists
    # in the same module scope.
    if type(data).__name__ == 'DataContainer':
        inner = data.data
        # inner is [[...], [...], ...] – one list per GPU batch group
        # For single-GPU eval, take index 0.
        if isinstance(inner, list) and len(inner) > 0:
            inner = inner[0]
        return _scatter_data(inner, device)
    elif isinstance(data, _Map):
        return {k: _scatter_data(v, device) for k, v in data.items()}
    elif isinstance(data, _t.Tensor):
        return data.to(device, non_blocking=True)
    elif isinstance(data, _Seq) and not isinstance(data, (str, bytes)):
        scattered = [_scatter_data(v, device) for v in data]
        return type(data)(scattered) if isinstance(data, tuple) else scattered
    else:
        return data


class MMDistributedDataParallel(_MMDDPBase):
    """mmcv 1.x MMDistributedDataParallel replacement.

    Extends mmengine's DDP to unwrap mmcv 1.x DataContainer objects in kwargs
    before forwarding them to the underlying module, replicating the scatter
    behaviour that mmcv 1.x MMDistributedDataParallel provided.
    """

    def forward(self, *args, **kwargs):
        device = next(self.parameters()).device
        kwargs = _scatter_data(kwargs, device)
        return super().forward(*args, **kwargs)


class DataContainer:
    """Full mmcv 1.x DataContainer replacement.

    Supports subscript access, len, iteration, and the _data alias used by
    some pipeline transforms so that model forward() code which does
    img_metas[0][0]['key'] works correctly after collation.
    """

    def __init__(self, data, stack=False, padding_value=0,
                 cpu_only=False, pad_dims=2):
        self.data = data
        self.stack = stack
        self.padding_value = padding_value
        self.cpu_only = cpu_only
        self.pad_dims = pad_dims

    # allow DC[i] → self.data[i]
    def __getitem__(self, idx):
        return self.data[idx]

    def __setitem__(self, idx, value):
        self.data[idx] = value

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data)

    # alias used in training pipeline code: example['gt_labels_3d']._data
    @property
    def _data(self):
        return self.data

    def __repr__(self):
        return f'DataContainer({self.data!r})'


def scatter(inputs, target_gpus, dim=0):
    """Stub scatter – returns inputs unchanged (single-GPU / CPU path)."""
    return inputs


from torch.utils.data.dataloader import default_collate as _default_collate  # noqa: E402
import torch.nn.functional as _F_collate                                      # noqa: E402
from collections.abc import Mapping as _Mapping, Sequence as _Sequence        # noqa: E402


def collate(batch, samples_per_gpu=1):
    """Full mmcv 1.x DataContainer-aware collate function.

    Replicates the behaviour of mmcv<=1.7 mmcv.parallel.collate:
    * DataContainer(cpu_only=True)  → DC([[s0,s1,...], [s2,s3,...]], cpu_only)
    * DataContainer(stack=True)     → DC([stacked_gpu0, stacked_gpu1, ...])
    * DataContainer(stack=False)    → DC([[s0,s1,...], [s2,s3,...]])
    * Mapping                       → recurse per-key
    * Sequence (non-str)            → zip + recurse
    * everything else               → default_collate
    """
    if not isinstance(batch, _Sequence):
        raise TypeError(f'batch must be a sequence, got {type(batch)}')

    elem = batch[0]

    if isinstance(elem, DataContainer):
        stacked = []
        if elem.cpu_only:
            for i in range(0, len(batch), samples_per_gpu):
                stacked.append(
                    [sample.data for sample in batch[i:i + samples_per_gpu]])
            return DataContainer(stacked, elem.stack, elem.padding_value,
                                 cpu_only=True)
        elif elem.stack:
            for i in range(0, len(batch), samples_per_gpu):
                chunk = batch[i:i + samples_per_gpu]
                if chunk[0].pad_dims is not None:
                    pad_dims = chunk[0].pad_dims
                    ndim = chunk[0].data.dim()
                    assert ndim > pad_dims
                    max_shape = [0] * pad_dims
                    for s in chunk:
                        for d in range(pad_dims):
                            max_shape[d] = max(max_shape[d], s.data.size(-d - 1))
                    padded = []
                    for s in chunk:
                        pad = []
                        for d in range(pad_dims):
                            pad = [0, max_shape[d] - s.data.size(-d - 1)] + pad
                        padded.append(
                            _F_collate.pad(s.data, pad, value=s.padding_value))
                    stacked.append(_default_collate(padded))
                else:
                    stacked.append(
                        _default_collate([s.data for s in chunk]))
            return DataContainer(stacked, elem.stack, elem.padding_value)
        else:
            for i in range(0, len(batch), samples_per_gpu):
                stacked.append(
                    [sample.data for sample in batch[i:i + samples_per_gpu]])
            return DataContainer(stacked, elem.stack, elem.padding_value)

    elif isinstance(elem, _Mapping):
        return {
            key: collate([d[key] for d in batch], samples_per_gpu)
            for key in elem
        }

    elif isinstance(elem, _Sequence) and not isinstance(elem, str):
        transposed = list(zip(*batch))
        return [collate(samples, samples_per_gpu) for samples in transposed]

    else:
        return _default_collate(batch)


_mmcv_parallel = _make_module(
    'mmcv.parallel',
    MMDataParallel=MMDataParallel,
    MMDistributedDataParallel=MMDistributedDataParallel,
    DataContainer=DataContainer,
    scatter=scatter,
    collate=collate,
)


# ---------------------------------------------------------------------------
# 3. mmcv.cnn.bricks.registry
#    ATTENTION / FEEDFORWARD_NETWORK / TRANSFORMER_LAYER* all map to
#    mmengine.registry.MODELS (root), because mmcv 2.x's build_attention()
#    internally uses mmengine.registry.MODELS for lookup.
# ---------------------------------------------------------------------------

from mmengine.registry import MODELS as _MMENGINE_MODELS    # noqa: E402

# Wrap with _ForceRegistry so UniV2X can re-register transformer/attention
# classes that might already exist in mmengine/mmdet built-in registries.
_MMENGINE_MODELS_FORCE = _ForceRegistry(_MMENGINE_MODELS)

_mmcv_cnn_bricks_registry = _make_module(
    'mmcv.cnn.bricks.registry',
    ATTENTION=_MMENGINE_MODELS_FORCE,
    FEEDFORWARD_NETWORK=_MMENGINE_MODELS_FORCE,
    TRANSFORMER_LAYER=_MMENGINE_MODELS_FORCE,
    TRANSFORMER_LAYER_SEQUENCE=_MMENGINE_MODELS_FORCE,
    CONV_LAYERS=_MMENGINE_MODELS_FORCE,
    PLUGIN_LAYERS=_MMENGINE_MODELS_FORCE,
    NORM_LAYERS=_MMENGINE_MODELS_FORCE,
    ACTIVATION_LAYERS=_MMENGINE_MODELS_FORCE,
)

# Make mmcv.cnn.bricks a proper package node so sub-module lookup works
import mmcv.cnn.bricks as _mmcv_cnn_bricks                  # noqa: E402
_mmcv_cnn_bricks.registry = _mmcv_cnn_bricks_registry


# ---------------------------------------------------------------------------
# 4. mmcv top-level: Config, DictAction, load, dump, mkdir_or_exist
# ---------------------------------------------------------------------------

from mmengine.config import Config, DictAction, ConfigDict   # noqa: E402
from mmengine.fileio import load as _fileio_load             # noqa: E402
from mmengine.fileio import dump as _fileio_dump             # noqa: E402
from mmengine.utils.path import mkdir_or_exist               # noqa: E402

import mmcv as _mmcv                                         # noqa: E402
_mmcv.Config = Config
_mmcv.ConfigDict = ConfigDict
_mmcv.DictAction = DictAction
_mmcv.load = _fileio_load
_mmcv.dump = _fileio_dump
_mmcv.mkdir_or_exist = mkdir_or_exist

# FileClient was removed from mmcv 2.x; re-export from mmengine
from mmengine.fileio import FileClient as _FileClient         # noqa: E402
_mmcv.FileClient = _FileClient

# ProgressBar was removed from mmcv 2.x; re-export from mmengine
from mmengine.utils import ProgressBar as _ProgressBar        # noqa: E402
_mmcv.ProgressBar = _ProgressBar

# track_iter_progress / track_parallel_progress were removed from mmcv 2.x
def _track_iter_progress(tasks, bar_width=50, file=None):  # noqa: ARG001 bar_width kept for API compat
    """Yield items from tasks while showing a progress bar (mmcv 1.x shim)."""
    import sys as _sys
    _file = file or _sys.stdout
    if hasattr(tasks, '__len__'):
        bar = _ProgressBar(len(tasks), file=_file)
        for t in tasks:
            yield t
            bar.update()
        _file.write('\n')
        _file.flush()
    else:
        for t in tasks:
            yield t

_mmcv.track_iter_progress = _track_iter_progress


# NuScenesDataset.DefaultAttribute was removed in mmdet3d 2.x; add it back.
from mmdet3d.datasets import NuScenesDataset as _NuScenesDataset   # noqa: E402
_NuScenesDataset.DefaultAttribute = {
    'car':                  'vehicle.parked',
    'pedestrian':           'pedestrian.moving',
    'trailer':              'vehicle.parked',
    'truck':                'vehicle.parked',
    'bus':                  'vehicle.moving',
    'motorcycle':           'cycle.without_rider',
    'construction_vehicle': 'vehicle.parked',
    'bicycle':              'cycle.without_rider',
    'barrier':              '',
    'traffic_cone':         '',
}


# ---------------------------------------------------------------------------
# 5. mmcv.utils: build_from_cfg, ConfigDict
# ---------------------------------------------------------------------------

from mmengine.registry import build_from_cfg, Registry       # noqa: E402

import torch as _torch_ver                                   # noqa: E402
import mmcv.utils as _mmcv_utils                            # noqa: E402
_mmcv_utils.build_from_cfg = build_from_cfg
_mmcv_utils.ConfigDict = ConfigDict
_mmcv_utils.Registry = Registry
# Store as the VERSION STRING (not tuple) so that digit_version() can parse it.
_mmcv_utils.TORCH_VERSION = _torch_ver.__version__


def _digit_version(version_str):
    """Convert version string like '2.1.0' OR tuple (2,1) to tuple (2, 1, 0)."""
    if isinstance(version_str, (tuple, list)):
        return tuple(version_str)
    # Strip any non-numeric suffix like '+cu118'
    clean = version_str.split('+')[0].split('a')[0].split('b')[0].split('rc')[0]
    parts = []
    for x in clean.split('.')[:3]:
        try:
            parts.append(int(x))
        except ValueError:
            break
    return tuple(parts)


_mmcv_utils.digit_version = _digit_version

from mmengine.utils import deprecated_api_warning, to_2tuple   # noqa: E402
_mmcv_utils.deprecated_api_warning = deprecated_api_warning
_mmcv_utils.to_2tuple = to_2tuple

# mmcv.utils.registry sub-module (from mmcv 1.x, now in mmengine)
_mmcv_utils_registry = _make_module(
    'mmcv.utils.registry',
    Registry=Registry,
    build_from_cfg=build_from_cfg,
)
_mmcv_utils.registry = _mmcv_utils_registry


# ---------------------------------------------------------------------------
# 6. mmdet.models.builder
#    HEADS / DETECTORS / NECKS / BACKBONES / LOSSES → mmdet.registry.MODELS
# ---------------------------------------------------------------------------

from mmdet.registry import MODELS as _MMDET_MODELS           # noqa: E402
from mmdet.registry import TASK_UTILS as _MMDET_TASK_UTILS   # noqa: E402

# Wrap with _ForceRegistry so UniV2X can re-register classes that already exist
# in mmdet 3.x (e.g. DiceLoss, SinePositionalEncoding) without KeyError.
_MMDET_MODELS_FORCE = _ForceRegistry(_MMDET_MODELS)


def _build_loss(cfg, **kwargs):
    return _MMDET_MODELS.build(cfg, **kwargs)

def _build_backbone(cfg, **kwargs):
    return _MMDET_MODELS.build(cfg, **kwargs)

def _build_head(cfg, **kwargs):
    return _MMDET_MODELS.build(cfg, **kwargs)

def _build_neck(cfg, **kwargs):
    return _MMDET_MODELS.build(cfg, **kwargs)

def _build_detector(cfg, **kwargs):
    return _MMDET_MODELS.build(cfg, **kwargs)

def _build_roi_extractor(cfg, **kwargs):
    return _MMDET_MODELS.build(cfg, **kwargs)


_mmdet_models_builder = _make_module(
    'mmdet.models.builder',
    HEADS=_MMDET_MODELS_FORCE,
    DETECTORS=_MMDET_MODELS_FORCE,
    NECKS=_MMDET_MODELS_FORCE,
    BACKBONES=_MMDET_MODELS_FORCE,
    LOSSES=_MMDET_MODELS_FORCE,
    ROI_EXTRACTORS=_MMDET_MODELS_FORCE,
    SHARED_HEADS=_MMDET_MODELS_FORCE,
    build_loss=_build_loss,
    build_backbone=_build_backbone,
    build_head=_build_head,
    build_neck=_build_neck,
    build_detector=_build_detector,
    build_roi_extractor=_build_roi_extractor,
)

# Attach to mmdet.models namespace (also expose registry aliases directly)
import mmdet.models as _mmdet_models                         # noqa: E402
_mmdet_models.builder = _mmdet_models_builder
_mmdet_models.HEADS = _MMDET_MODELS_FORCE
_mmdet_models.DETECTORS = _MMDET_MODELS_FORCE
_mmdet_models.NECKS = _MMDET_MODELS_FORCE
_mmdet_models.BACKBONES = _MMDET_MODELS_FORCE
_mmdet_models.LOSSES = _MMDET_MODELS_FORCE
_mmdet_models.ROI_EXTRACTORS = _MMDET_MODELS_FORCE
_mmdet_models.build_loss = _build_loss


# ---------------------------------------------------------------------------
# 7. mmdet.core
# ---------------------------------------------------------------------------

from mmdet.models.utils.misc import multi_apply              # noqa: E402
from mmdet.utils.dist_utils import reduce_mean               # noqa: E402
from mmdet.structures.bbox.transforms import (               # noqa: E402
    bbox2result, bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh,
)
from mmdet.structures.bbox import bbox_overlaps              # noqa: E402
from mmdet.models.task_modules.builder import (              # noqa: E402
    build_anchor_generator, build_assigner, build_sampler, build_match_cost,
)
from mmdet.models.task_modules.prior_generators.utils import (  # noqa: E402
    anchor_inside_flags,
)

# Assigners
from mmdet.models.task_modules.assigners import AssignResult, BaseAssigner  # noqa: E402
from mmdet.models.task_modules.assigners.assign_result import AssignResult as _AssignResult  # noqa: E402
from mmdet.models.task_modules.assigners.base_assigner import BaseAssigner as _BaseAssigner  # noqa: E402

# Samplers
from mmdet.models.task_modules.samplers import (             # noqa: E402
    BaseSampler, RandomSampler, SamplingResult,
)
from mmdet.models.task_modules.samplers.base_sampler import BaseSampler as _BaseSampler  # noqa: E402
from mmdet.models.task_modules.samplers.random_sampler import RandomSampler as _RandomSampler  # noqa: E402
from mmdet.models.task_modules.samplers.sampling_result import SamplingResult as _SamplingResult  # noqa: E402

# Coders
from mmdet.models.task_modules.coders.base_bbox_coder import BaseBBoxCoder  # noqa: E402

# Match costs
from mmdet.registry import TASK_UTILS as _MMDET_TASK_UTILS_CORE  # noqa: E402

# Mask module (used as `from mmdet.core import mask`)
import mmdet.structures.mask as _mmdet_mask_mod             # noqa: E402

# EvalHook stubs (mmdet 1.x hooks, removed in 3.x)
class _EvalHookStub:
    """Stub for mmdet.core EvalHook (removed in mmdet 3.x)."""
    def __init__(self, *args, **kwargs): pass

class _DistEvalHookStub(_EvalHookStub):
    pass


_mmdet_core = _make_module(
    'mmdet.core',
    multi_apply=multi_apply,
    reduce_mean=reduce_mean,
    bbox2result=bbox2result,
    bbox_overlaps=bbox_overlaps,
    bbox_cxcywh_to_xyxy=bbox_cxcywh_to_xyxy,
    bbox_xyxy_to_cxcywh=bbox_xyxy_to_cxcywh,
    build_anchor_generator=build_anchor_generator,
    anchor_inside_flags=anchor_inside_flags,
    build_assigner=build_assigner,
    build_sampler=build_sampler,
    build_match_cost=build_match_cost,
    AssignResult=_AssignResult,
    BaseAssigner=_BaseAssigner,
    BaseSampler=_BaseSampler,
    RandomSampler=_RandomSampler,
    SamplingResult=_SamplingResult,
    BaseBBoxCoder=BaseBBoxCoder,
    EvalHook=_EvalHookStub,
    DistEvalHook=_DistEvalHookStub,
    mask=_mmdet_mask_mod,
)

# --- mmdet.core.bbox sub-module and its children ---

_mmdet_core_bbox_assigners = _make_module(
    'mmdet.core.bbox.assigners',
    AssignResult=_AssignResult,
    BaseAssigner=_BaseAssigner,
    assign_result=types.ModuleType('mmdet.core.bbox.assigners.assign_result'),
    base_assigner=types.ModuleType('mmdet.core.bbox.assigners.base_assigner'),
)
setattr(_mmdet_core_bbox_assigners.assign_result, 'AssignResult', _AssignResult)
setattr(_mmdet_core_bbox_assigners.base_assigner, 'BaseAssigner', _BaseAssigner)
sys.modules['mmdet.core.bbox.assigners'] = _mmdet_core_bbox_assigners
sys.modules['mmdet.core.bbox.assigners.assign_result'] = _mmdet_core_bbox_assigners.assign_result
sys.modules['mmdet.core.bbox.assigners.base_assigner'] = _mmdet_core_bbox_assigners.base_assigner

_mmdet_core_bbox_samplers_result = types.ModuleType('mmdet.core.bbox.samplers.sampling_result')
setattr(_mmdet_core_bbox_samplers_result, 'SamplingResult', _SamplingResult)
_mmdet_core_bbox_samplers_rand = types.ModuleType('mmdet.core.bbox.samplers.random_sampler')
setattr(_mmdet_core_bbox_samplers_rand, 'RandomSampler', _RandomSampler)
_mmdet_core_bbox_samplers_base = types.ModuleType('mmdet.core.bbox.samplers.base_sampler')
setattr(_mmdet_core_bbox_samplers_base, 'BaseSampler', _BaseSampler)

_mmdet_core_bbox_samplers = _make_module(
    'mmdet.core.bbox.samplers',
    BaseSampler=_BaseSampler,
    RandomSampler=_RandomSampler,
    SamplingResult=_SamplingResult,
    base_sampler=_mmdet_core_bbox_samplers_base,
    random_sampler=_mmdet_core_bbox_samplers_rand,
    sampling_result=_mmdet_core_bbox_samplers_result,
)
sys.modules['mmdet.core.bbox.samplers.base_sampler'] = _mmdet_core_bbox_samplers_base
sys.modules['mmdet.core.bbox.samplers.random_sampler'] = _mmdet_core_bbox_samplers_rand
sys.modules['mmdet.core.bbox.samplers.sampling_result'] = _mmdet_core_bbox_samplers_result

# Use mmengine.TASK_UTILS (root) for registries that UniV2X's custom modules
# register into, so they don't clash with mmdet 3.x pre-registered ones
# (e.g. DiceCost is already in mmdet.TASK_UTILS).
# mmdet.TASK_UTILS has mmengine.TASK_UTILS as parent, so build() still finds
# custom classes registered in the parent.
from mmengine.registry import TASK_UTILS as _MMENGINE_TASK_UTILS  # noqa: E402
_MMENGINE_TASK_UTILS_FORCE = _ForceRegistry(_MMENGINE_TASK_UTILS)

_mmdet_core_bbox_match_costs_builder = _make_module(
    'mmdet.core.bbox.match_costs.builder',
    MATCH_COST=_MMENGINE_TASK_UTILS_FORCE,
    build_match_cost=build_match_cost,
)
_mmdet_core_bbox_match_costs = _make_module(
    'mmdet.core.bbox.match_costs',
    build_match_cost=build_match_cost,
    MATCH_COST=_MMENGINE_TASK_UTILS_FORCE,
    builder=_mmdet_core_bbox_match_costs_builder,
)
sys.modules['mmdet.core.bbox.match_costs.builder'] = _mmdet_core_bbox_match_costs_builder

_mmdet_core_bbox_builder = _make_module(
    'mmdet.core.bbox.builder',
    BBOX_ASSIGNERS=_MMENGINE_TASK_UTILS_FORCE,
    BBOX_CODERS=_MMENGINE_TASK_UTILS_FORCE,
    BBOX_SAMPLERS=_MMENGINE_TASK_UTILS_FORCE,
    MATCH_COST=_MMENGINE_TASK_UTILS_FORCE,
)

_mmdet_core_bbox_transforms = _make_module(
    'mmdet.core.bbox.transforms',
    bbox2result=bbox2result,
    bbox_cxcywh_to_xyxy=bbox_cxcywh_to_xyxy,
    bbox_xyxy_to_cxcywh=bbox_xyxy_to_cxcywh,
    bbox_overlaps=bbox_overlaps,
)
sys.modules['mmdet.core.bbox.transforms'] = _mmdet_core_bbox_transforms

_mmdet_core_bbox = _make_module(
    'mmdet.core.bbox',
    bbox2result=bbox2result,
    bbox_overlaps=bbox_overlaps,
    bbox_cxcywh_to_xyxy=bbox_cxcywh_to_xyxy,
    bbox_xyxy_to_cxcywh=bbox_xyxy_to_cxcywh,
    build_anchor_generator=build_anchor_generator,
    anchor_inside_flags=anchor_inside_flags,
    AssignResult=_AssignResult,
    BaseAssigner=_BaseAssigner,
    BaseBBoxCoder=BaseBBoxCoder,
    build_match_cost=build_match_cost,
    assigners=_mmdet_core_bbox_assigners,
    samplers=_mmdet_core_bbox_samplers,
    match_costs=_mmdet_core_bbox_match_costs,
    builder=_mmdet_core_bbox_builder,
    transforms=_mmdet_core_bbox_transforms,
)
_mmdet_core.bbox = _mmdet_core_bbox

# --- mmdet.core.evaluation sub-module ---
_mmdet_core_eval_hooks = _make_module(
    'mmdet.core.evaluation.eval_hooks',
    EvalHook=_EvalHookStub,
    DistEvalHook=_DistEvalHookStub,
)
_mmdet_core_evaluation = _make_module(
    'mmdet.core.evaluation',
    EvalHook=_EvalHookStub,
    DistEvalHook=_DistEvalHookStub,
    eval_hooks=_mmdet_core_eval_hooks,
)
sys.modules['mmdet.core.evaluation.eval_hooks'] = _mmdet_core_eval_hooks
_mmdet_core.evaluation = _mmdet_core_evaluation

# --- mmdet.core.mask sub-module ---
_mmdet_core.mask = _mmdet_mask_mod


# ---------------------------------------------------------------------------
# 8. mmdet.apis: set_random_seed
# ---------------------------------------------------------------------------

import mmdet.apis as _mmdet_apis                             # noqa: E402
_mmdet_apis.set_random_seed = set_random_seed


# ---------------------------------------------------------------------------
# 9. mmdet3d.core and sub-modules
# ---------------------------------------------------------------------------

from mmdet3d.structures import (                             # noqa: E402
    Box3DMode, Coord3DMode,
    LiDARInstance3DBoxes, BaseInstance3DBoxes,
    CameraInstance3DBoxes, DepthInstance3DBoxes,
    bbox3d2result,
    BboxOverlaps3D, xywhr2xyxyr,
)
from mmdet3d.structures.points import BasePoints, get_points_type  # noqa: E402
from mmdet3d.structures.ops.iou3d_calculator import (        # noqa: E402
    BboxOverlaps3D as _BboxOverlaps3D_iou,
)
from mmdet3d.structures.ops import (                         # noqa: E402
    bbox_overlaps_nearest_3d as _bbox_overlaps_nearest_3d,
    bbox_overlaps_3d as _bbox_overlaps_3d,
    BboxOverlapsNearest3D as _BboxOverlapsNearest3D,
)
from mmdet3d.models.test_time_augs.merge_augs import (       # noqa: E402
    merge_aug_bboxes_3d,
)

# mmdet3d.core.bbox.iou_calculators.iou3d_calculator sub-module
_mmdet3d_core_bbox_iou_calc = _make_module(
    'mmdet3d.core.bbox.iou_calculators.iou3d_calculator',
    BboxOverlaps3D=_BboxOverlaps3D_iou,
    BboxOverlapsNearest3D=_BboxOverlapsNearest3D,
    bbox_overlaps_nearest_3d=_bbox_overlaps_nearest_3d,
    bbox_overlaps_3d=_bbox_overlaps_3d,
)

# mmdet3d.core.bbox.iou_calculators  (used directly in some files)
_mmdet3d_core_bbox_iou = _make_module(
    'mmdet3d.core.bbox.iou_calculators',
    BboxOverlaps3D=_BboxOverlaps3D_iou,
    BboxOverlapsNearest3D=_BboxOverlapsNearest3D,
    bbox_overlaps_nearest_3d=_bbox_overlaps_nearest_3d,
    bbox_overlaps_3d=_bbox_overlaps_3d,
)
_mmdet3d_core_bbox_iou.iou3d_calculator = _mmdet3d_core_bbox_iou_calc

# mmdet3d.core.bbox  (also used for coders lookup)
from mmdet3d.models.task_modules.builder import build_bbox_coder  # noqa: E402
from mmdet3d.registry import TASK_UTILS as _MMDET3D_TASK_UTILS    # noqa: E402

def _mmdet3d_build_bbox_coder(cfg, **kwargs):
    return build_bbox_coder(cfg, **kwargs)

_mmdet3d_core_bbox = _make_module(
    'mmdet3d.core.bbox',
    Box3DMode=Box3DMode,
    Coord3DMode=Coord3DMode,
    LiDARInstance3DBoxes=LiDARInstance3DBoxes,
    BaseInstance3DBoxes=BaseInstance3DBoxes,
    CameraInstance3DBoxes=CameraInstance3DBoxes,
    DepthInstance3DBoxes=DepthInstance3DBoxes,
    BboxOverlaps3D=_BboxOverlaps3D_iou,
    build_bbox_coder=_mmdet3d_build_bbox_coder,
    iou_calculators=_mmdet3d_core_bbox_iou,
)
# also expose coders sub-module
_mmdet3d_core_bbox_coders = _make_module(
    'mmdet3d.core.bbox.coders',
    build_bbox_coder=_mmdet3d_build_bbox_coder,
)
_mmdet3d_core_bbox.coders = _mmdet3d_core_bbox_coders

# mmdet3d.core.points
_mmdet3d_core_points = _make_module(
    'mmdet3d.core.points',
    BasePoints=BasePoints,
    get_points_type=get_points_type,
)

# mmdet3d.core  (top-level)
_mmdet3d_core = _make_module(
    'mmdet3d.core',
    Box3DMode=Box3DMode,
    Coord3DMode=Coord3DMode,
    LiDARInstance3DBoxes=LiDARInstance3DBoxes,
    BaseInstance3DBoxes=BaseInstance3DBoxes,
    CameraInstance3DBoxes=CameraInstance3DBoxes,
    DepthInstance3DBoxes=DepthInstance3DBoxes,
    bbox3d2result=bbox3d2result,
    BboxOverlaps3D=_BboxOverlaps3D_iou,
    xywhr2xyxyr=xywhr2xyxyr,
    merge_aug_bboxes_3d=merge_aug_bboxes_3d,
    build_bbox_coder=_mmdet3d_build_bbox_coder,
    bbox=_mmdet3d_core_bbox,
    points=_mmdet3d_core_points,
)


# ---------------------------------------------------------------------------
# 10. mmdet3d.models.builder + mmdet3d.models.build_model
# ---------------------------------------------------------------------------

from mmdet3d.registry import MODELS as _MMDET3D_MODELS       # noqa: E402

# Wrap so UniV2X classes can re-register without conflict
_MMDET3D_MODELS_FORCE = _ForceRegistry(_MMDET3D_MODELS)


def _mmdet3d_build_backbone(cfg, **kwargs):
    return _MMDET3D_MODELS.build(cfg, **kwargs)

def _mmdet3d_build_head(cfg, **kwargs):
    return _MMDET3D_MODELS.build(cfg, **kwargs)

def _mmdet3d_build_neck(cfg, **kwargs):
    return _MMDET3D_MODELS.build(cfg, **kwargs)

def _mmdet3d_build_model(cfg, train_cfg=None, test_cfg=None):
    if train_cfg is not None or test_cfg is not None:
        cfg = cfg.copy()
        if train_cfg is not None:
            cfg['train_cfg'] = train_cfg
        if test_cfg is not None:
            cfg['test_cfg'] = test_cfg
    model_type = cfg.get('type', '') if isinstance(cfg, dict) else getattr(cfg, 'type', '')
    if model_type in _MMDET3D_MODELS._module_dict:
        return _MMDET3D_MODELS.build(cfg)
    # Fall back: check mmdet DETECTORS registry (plugin models register there)
    from mmdet.registry import MODELS as _MMDET_MODELS_REG
    if model_type in _MMDET_MODELS_REG._module_dict:
        return _MMDET_MODELS_REG.build(cfg)
    return _MMDET3D_MODELS.build(cfg)  # let it raise the descriptive error


_mmdet3d_models_builder = _make_module(
    'mmdet3d.models.builder',
    MODELS=_MMDET3D_MODELS_FORCE,
    build_backbone=_mmdet3d_build_backbone,
    build_head=_mmdet3d_build_head,
    build_neck=_mmdet3d_build_neck,
    build_model=_mmdet3d_build_model,
)

# Patch mmdet3d.models namespace directly
import mmdet3d.models as _mmdet3d_models                     # noqa: E402
_mmdet3d_models.builder = _mmdet3d_models_builder
_mmdet3d_models.build_model = _mmdet3d_build_model


# ---------------------------------------------------------------------------
# 10b. Patch MVXTwoStageDetector's internal MODELS to a combined registry
#      that searches mmdet3d first, then mmdet. This lets it build plugin
#      heads/necks/backbones that are registered via mmdet.models.HEADS etc.
# ---------------------------------------------------------------------------

class _CombinedModelRegistry:
    """Falls back to mmdet's MODELS when a type is absent from mmdet3d's."""

    def __init__(self, primary, secondary):
        self._primary = primary
        self._secondary = secondary

    def build(self, cfg, **kwargs):
        type_name = (cfg.get('type', '') if isinstance(cfg, dict)
                     else getattr(cfg, 'type', ''))
        if type_name in self._primary._module_dict:
            return self._primary.build(cfg, **kwargs)
        from mmdet.registry import MODELS as _MD
        if type_name in _MD._module_dict:
            return _MD.build(cfg, **kwargs)
        return self._primary.build(cfg, **kwargs)   # raises descriptive error

    def register_module(self, name=None, force=False, module=None):  # noqa: ARG002
        return self._primary.register_module(name=name, force=True, module=module)

    def __getattr__(self, name):
        return getattr(self._primary, name)

    def __contains__(self, item):
        return item in self._primary

    def get(self, key):
        return self._primary.get(key)


_COMBINED_MODELS = _CombinedModelRegistry(_MMDET3D_MODELS, _MMDET_MODELS)

# Patch MVXTwoStageDetector's module-level MODELS reference
try:
    import mmdet3d.models.detectors.mvx_two_stage as _mvx_mod
    _mvx_mod.MODELS = _COMBINED_MODELS
except Exception:
    pass


# ---------------------------------------------------------------------------
# 11. mmdet3d.apis.single_gpu_test  +  mmdet3d.datasets.build_dataset
# ---------------------------------------------------------------------------

import torch                                                 # noqa: E402
import mmcv as _mmcv2                                        # noqa: E402


def single_gpu_test(model, data_loader, show=False, out_dir=None,
                    show_score_thr=0.3):
    """Minimal single-GPU test loop compatible with mmdet3d 0.x callers."""
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = _mmcv2.ProgressBar(len(dataset))
    for data in data_loader:
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
        if isinstance(result, list):
            results.extend(result)
        else:
            results.append(result)
        batch_size = len(next(iter(data.values())))
        for _ in range(batch_size):
            prog_bar.update()
    return results


from mmdet3d.registry import DATASETS as _MMDET3D_DATASETS   # noqa: E402
from mmdet3d.registry import TRANSFORMS as _MMDET3D_TRANSFORMS  # noqa: E402
from mmengine.registry import TRANSFORMS as _MMENGINE_TRANSFORMS_DS  # noqa: E402
import mmdet3d.datasets as _mmdet3d_datasets_mod             # noqa: E402  # trigger registration
import mmdet3d.datasets.transforms as _mmdet3d_transforms_mod  # noqa: E402, F401  # trigger transform registration

# Promote all mmdet3d transforms into mmengine's TRANSFORMS registry so
# that BaseDataset.Compose (which uses mmengine.TRANSFORMS) can find them.
for _name, _cls in list(_MMDET3D_TRANSFORMS._module_dict.items()):
    if _name not in _MMENGINE_TRANSFORMS_DS._module_dict:
        _MMENGINE_TRANSFORMS_DS.register_module(module=_cls, name=_name, force=True)


# ---------------------------------------------------------------------------
# 11b. DefaultFormatBundle3D  (removed from mmdet3d 2.x; reconstruct from 1.x)
#      Also expose mmdet3d.datasets.pipelines as compat shim module.
# ---------------------------------------------------------------------------

from mmdet3d.structures.points import BasePoints as _BasePoints  # noqa: E402
from mmcv.parallel import DataContainer as _DC_dfb               # noqa: E402


class DefaultFormatBundle3D:
    """Compat shim reproducing mmdet3d 0.x DefaultFormatBundle3D behavior."""

    def __init__(self, class_names, with_gt=True, with_label=True):
        self.class_names = class_names
        self.with_gt = with_gt
        self.with_label = with_label

    def __call__(self, results):
        import numpy as np
        from mmcv import to_tensor as _t2t

        if 'points' in results:
            assert isinstance(results['points'], _BasePoints)
            results['points'] = _DC_dfb(results['points'].tensor)

        for key in ['voxels', 'coors', 'voxel_centers', 'num_points']:
            if key not in results:
                continue
            results[key] = _DC_dfb(_t2t(results[key]), stack=False)

        if self.with_gt:
            if 'gt_bboxes_3d_mask' in results:
                mask = results['gt_bboxes_3d_mask']
                results['gt_bboxes_3d'] = results['gt_bboxes_3d'][mask]
                if 'gt_names_3d' in results:
                    results['gt_names_3d'] = results['gt_names_3d'][mask]
                if 'centers2d' in results:
                    results['centers2d'] = results['centers2d'][mask]
                if 'depths' in results:
                    results['depths'] = results['depths'][mask]
            if 'gt_bboxes_mask' in results:
                mask = results['gt_bboxes_mask']
                if 'gt_bboxes' in results:
                    results['gt_bboxes'] = results['gt_bboxes'][mask]
                results['gt_names'] = results['gt_names'][mask]
            if self.with_label:
                if 'gt_names' in results and len(results['gt_names']) == 0:
                    results['gt_labels'] = np.array([], dtype=np.int64)
                    results['attr_labels'] = np.array([], dtype=np.int64)
                elif 'gt_names' in results and isinstance(results['gt_names'][0], list):
                    results['gt_labels'] = [
                        np.array([self.class_names.index(n) for n in res], dtype=np.int64)
                        for res in results['gt_names']
                    ]
                elif 'gt_names' in results:
                    results['gt_labels'] = np.array(
                        [self.class_names.index(n) for n in results['gt_names']], dtype=np.int64)
                if 'gt_names_3d' in results:
                    results['gt_labels_3d'] = np.array(
                        [self.class_names.index(n) for n in results['gt_names_3d']], dtype=np.int64)

        # Run the base formatting (handles img, gt_bboxes_3d, gt_labels, etc.)
        if 'img' in results:
            if isinstance(results['img'], list):
                imgs = [img.transpose(2, 0, 1) for img in results['img']]
                imgs = np.ascontiguousarray(np.stack(imgs, axis=0))
                results['img'] = _DC_dfb(_t2t(imgs), stack=True)
            else:
                img = np.ascontiguousarray(results['img'].transpose(2, 0, 1))
                results['img'] = _DC_dfb(_t2t(img), stack=True)
        for key in ['proposals', 'gt_bboxes', 'gt_bboxes_ignore', 'gt_labels',
                    'gt_labels_3d', 'attr_labels', 'pts_instance_mask',
                    'pts_semantic_mask', 'centers2d', 'depths']:
            if key not in results:
                continue
            if isinstance(results[key], list):
                results[key] = _DC_dfb([_t2t(r) for r in results[key]])
            else:
                results[key] = _DC_dfb(_t2t(results[key]))
        if 'gt_bboxes_3d' in results:
            from mmdet3d.structures import BaseInstance3DBoxes as _B3D
            if isinstance(results['gt_bboxes_3d'], _B3D):
                results['gt_bboxes_3d'] = _DC_dfb(results['gt_bboxes_3d'], cpu_only=True)
            else:
                results['gt_bboxes_3d'] = _DC_dfb(_t2t(results['gt_bboxes_3d']))
        if 'gt_masks' in results:
            results['gt_masks'] = _DC_dfb(results['gt_masks'], cpu_only=True)
        if 'gt_semantic_seg' in results:
            results['gt_semantic_seg'] = _DC_dfb(
                _t2t(results['gt_semantic_seg'][None, ...]), stack=True)
        return results

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'class_names={self.class_names}, '
                f'with_gt={self.with_gt}, with_label={self.with_label})')


# Register DefaultFormatBundle3D in both mmdet3d and mmengine TRANSFORMS
_MMDET3D_TRANSFORMS.register_module(module=DefaultFormatBundle3D, name='DefaultFormatBundle3D', force=True)
_MMENGINE_TRANSFORMS_DS.register_module(module=DefaultFormatBundle3D, name='DefaultFormatBundle3D', force=True)

# Also expose via mmdet3d.datasets.pipelines shim
_mmdet3d_pipelines_shim = _make_module(
    'mmdet3d.datasets.pipelines',
    DefaultFormatBundle3D=DefaultFormatBundle3D,
)
sys.modules.setdefault('mmdet3d.datasets.pipelines', _mmdet3d_pipelines_shim)


def build_dataset(cfg, default_args=None):
    """Build mmdet3d dataset from config dict (mmdet3d 0.x API).

    Searches mmdet3d registry first, then mmdet registry. This is needed
    because custom datasets (e.g. SPDE2EDataset) register themselves via
    ``from mmdet.datasets import DATASETS`` which populates the mmdet registry
    rather than the mmdet3d one.
    """
    dataset_type = cfg.get('type', '') if isinstance(cfg, dict) else getattr(cfg, 'type', '')
    if dataset_type in _MMDET3D_DATASETS._module_dict:
        return _MMDET3D_DATASETS.build(cfg, default_args=default_args)
    # Fall back to mmdet registry (populated by @DATASETS.register_module() in plugin datasets)
    from mmdet.registry import DATASETS as _MMDET_DS
    if dataset_type in _MMDET_DS._module_dict:
        return _MMDET_DS.build(cfg, default_args=default_args)
    # Last resort: let mmdet3d raise the descriptive KeyError
    return _MMDET3D_DATASETS.build(cfg, default_args=default_args)


import mmdet3d.apis as _mmdet3d_apis                         # noqa: E402
_mmdet3d_apis.single_gpu_test = single_gpu_test

import mmdet3d.datasets as _mmdet3d_datasets                 # noqa: E402
_mmdet3d_datasets.build_dataset = build_dataset


# ---------------------------------------------------------------------------
# 12. mmdet.models.utils sub-modules
#     mmdet 2.x had builder/transformer here; mmdet 3.x removed them.
# ---------------------------------------------------------------------------

from mmdet.models.layers.transformer import inverse_sigmoid  # noqa: E402
from mmcv.cnn.bricks.transformer import (                    # noqa: E402
    build_transformer_layer_sequence,
)

# TRANSFORMER registry → mmengine.MODELS (same root used by build_attention)
_TRANSFORMER_registry = _MMENGINE_MODELS_FORCE


class _Transformer1xCompat(BaseModule):
    """mmdet 1.x Transformer base class.

    Accepts encoder= and decoder= config dicts, builds them with
    build_transformer_layer_sequence, and exposes self.encoder /
    self.decoder / self.embed_dims — exactly the API that plugin
    classes like SegDeformableTransformer rely on.
    """

    def __init__(self, encoder=None, decoder=None, init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        if encoder is not None:
            self.encoder = build_transformer_layer_sequence(encoder)
        if decoder is not None:
            self.decoder = build_transformer_layer_sequence(decoder)
        if hasattr(self, 'encoder'):
            self.embed_dims = self.encoder.embed_dims

    def init_weights(self):
        import torch.nn as _nn2
        for p in self.parameters():
            if p.dim() > 1:
                _nn2.init.xavier_uniform_(p)


# Register in TRANSFORMER / MODELS so build_transformer finds it
_TRANSFORMER_registry.register_module(module=_Transformer1xCompat, name='Transformer')


def _build_transformer(cfg, **kwargs):
    """build_transformer: build via mmengine MODELS registry (TRANSFORMER alias)."""
    return build_transformer_layer_sequence(cfg, **kwargs)


# mmdet.models.utils.transformer  (inverse_sigmoid lives here in mmdet 2.x)
_mmdet_utils_transformer = _make_module(
    'mmdet.models.utils.transformer',
    inverse_sigmoid=inverse_sigmoid,
    Transformer=_Transformer1xCompat,
)

# mmdet.models.utils.builder  (TRANSFORMER registry)
_mmdet_utils_builder = _make_module(
    'mmdet.models.utils.builder',
    TRANSFORMER=_TRANSFORMER_registry,
)

# mmdet.models.utils  (top-level, merges all)
_mmdet_utils = _make_module(
    'mmdet.models.utils',
    inverse_sigmoid=inverse_sigmoid,
    Transformer=_Transformer1xCompat,
    build_transformer=_build_transformer,
    multi_apply=multi_apply,
    transformer=_mmdet_utils_transformer,
    builder=_mmdet_utils_builder,
)

# Attach to mmdet.models namespace
_mmdet_models.utils = _mmdet_utils
sys.modules['mmdet.models.utils.transformer'] = _mmdet_utils_transformer
sys.modules['mmdet.models.utils.builder'] = _mmdet_utils_builder


# ---------------------------------------------------------------------------
# 13. mmcv.cnn.bricks.registry: POSITIONAL_ENCODING
#     mmdet 2.x had this; in mmcv 2.x positional encodings register via MODELS
# ---------------------------------------------------------------------------

_mmcv_cnn_bricks_registry.POSITIONAL_ENCODING = _MMENGINE_MODELS_FORCE
sys.modules['mmcv.cnn.bricks.registry'] = _mmcv_cnn_bricks_registry


# ---------------------------------------------------------------------------
# 14. mmdet.datasets: DATASETS, build_dataset, replace_ImageToTensor
# ---------------------------------------------------------------------------

from mmdet.registry import DATASETS as _MMDET_DATASETS       # noqa: E402


def _replace_ImageToTensor(pipelines):
    """Stub: replaces ImageToTensor with DefaultFormatBundle in test pipelines.
    In mmdet 3.x this is no longer needed as the pipeline changed."""
    return pipelines


def _mmdet_build_dataset(cfg, default_args=None):
    return _MMDET_DATASETS.build(cfg, default_args=default_args)


import mmdet.datasets as _mmdet_datasets_mod                  # noqa: E402
_mmdet_datasets_mod.DATASETS = _MMDET_DATASETS
_mmdet_datasets_mod.build_dataset = _mmdet_build_dataset
_mmdet_datasets_mod.replace_ImageToTensor = _replace_ImageToTensor

# mmdet.datasets.pipelines: to_tensor and other transforms (removed in mmdet 3.x)
from mmcv import to_tensor as _to_tensor                     # noqa: E402
from mmcv.transforms import ToTensor as _ToTensor            # noqa: E402

_mmdet_datasets_pipelines = _make_module(
    'mmdet.datasets.pipelines',
    to_tensor=_to_tensor,
    ToTensor=_ToTensor,
)
_mmdet_datasets_mod.pipelines = _mmdet_datasets_pipelines

# mmdet.datasets.samplers: GroupSampler (removed in mmdet 3.x)
import torch.utils.data as _torch_data                      # noqa: E402

class _GroupSampler(_torch_data.Sampler):
    """Stub GroupSampler – used only for non-distributed inference timing."""
    def __init__(self, dataset, samples_per_gpu=1):
        self.dataset = dataset
        self.samples_per_gpu = samples_per_gpu
        self.indices = list(range(len(dataset)))
    def __iter__(self):
        return iter(self.indices)
    def __len__(self):
        return len(self.dataset)

import mmdet.datasets.samplers as _mmdet_samplers            # noqa: E402
_mmdet_samplers.GroupSampler = _GroupSampler

# mmdet.datasets.builder: _concat_dataset (private helper, removed in 3.x)
from mmengine.dataset import ConcatDataset as _ConcatDataset  # noqa: E402

def _concat_dataset(cfg, default_args=None):
    ann_files = cfg['ann_file'] if isinstance(cfg.get('ann_file'), (list, tuple)) else [cfg.get('ann_file')]
    datasets = []
    for ann_file in ann_files:
        data_cfg = cfg.copy()
        data_cfg['ann_file'] = ann_file
        datasets.append(_MMDET_DATASETS.build(data_cfg, default_args=default_args))
    return _ConcatDataset(datasets)

from mmengine.registry import TRANSFORMS as _MMENGINE_TRANSFORMS  # noqa: E402

_mmdet_datasets_builder_mod = _make_module(
    'mmdet.datasets.builder',
    _concat_dataset=_concat_dataset,
    build_dataset=_mmdet_build_dataset,
    DATASETS=_MMDET_DATASETS,
    PIPELINES=_MMENGINE_TRANSFORMS,  # PIPELINES → TRANSFORMS in mmengine
)
_mmdet_datasets_mod.builder = _mmdet_datasets_builder_mod

# mmdet.datasets.dataset_wrappers: ClassBalancedDataset, ConcatDataset, RepeatDataset
from mmengine.dataset import RepeatDataset as _RepeatDataset  # noqa: E402
from mmengine.dataset import ClassBalancedDataset as _ClassBalancedDataset  # noqa: E402

import mmdet.datasets.dataset_wrappers as _mmdet_dataset_wrappers  # noqa: E402
_mmdet_dataset_wrappers.ClassBalancedDataset = _ClassBalancedDataset
_mmdet_dataset_wrappers.ConcatDataset = _ConcatDataset
_mmdet_dataset_wrappers.RepeatDataset = _RepeatDataset


# ---------------------------------------------------------------------------
# 15. mmcv.runner extended hooks + build_optimizer / build_runner
# ---------------------------------------------------------------------------

from mmengine.hooks import DistSamplerSeedHook                # noqa: E402
from mmengine.logging import MMLogger                         # noqa: E402


class _Fp16OptimizerHookStub:
    """Stub for Fp16OptimizerHook – training only, not needed for inference."""
    def __init__(self, *args, **kwargs): pass

class _OptimizerHookStub:
    """Stub for OptimizerHook – training only."""
    def __init__(self, *args, **kwargs): pass

class _GradientCumulativeFp16OptimizerHookStub:
    def __init__(self, *args, **kwargs): pass

class _GradientCumulativeOptimizerHookStub:
    def __init__(self, *args, **kwargs): pass


def _build_optimizer(model, cfg):
    """Stub build_optimizer – only needed at train time."""
    from mmengine.optim import build_optim_wrapper
    raise NotImplementedError(
        'build_optimizer: use mmengine.optim.build_optim_wrapper instead')

def _build_runner(cfg, default_args=None):
    raise NotImplementedError(
        'build_runner: use mmengine.runner.Runner instead')


# Add extended items to mmcv.runner shim
_mmcv_runner.DistSamplerSeedHook = DistSamplerSeedHook
_mmcv_runner.Fp16OptimizerHook = _Fp16OptimizerHookStub
_mmcv_runner.OptimizerHook = _OptimizerHookStub
_mmcv_runner.GradientCumulativeFp16OptimizerHook = _GradientCumulativeFp16OptimizerHookStub
_mmcv_runner.GradientCumulativeOptimizerHook = _GradientCumulativeOptimizerHookStub
_mmcv_runner.build_optimizer = _build_optimizer
_mmcv_runner.build_runner = _build_runner
_mmcv_runner.HOOKS = _MMENGINE_MODELS  # HOOKS registry → mmengine registry


# ---------------------------------------------------------------------------
# 16. mmdet.utils: get_root_logger
# ---------------------------------------------------------------------------

def _get_root_logger(log_file=None, log_level=20, name='mmdet'):
    """get_root_logger: returns an MMLogger instance (mmengine equivalent)."""
    return MMLogger.get_instance(name, log_file=log_file, log_level=log_level)


import mmdet.utils as _mmdet_utils_mod                       # noqa: E402
_mmdet_utils_mod.get_root_logger = _get_root_logger


# ---------------------------------------------------------------------------
# 17. mmdet3d.core.bbox.box_np_ops  (removed in mmdet3d 1.x; in structures now)
# ---------------------------------------------------------------------------

from mmdet3d.structures.bbox_3d.utils import points_cam2img  # noqa: E402

_mmdet3d_core_bbox_box_np_ops = _make_module(
    'mmdet3d.core.bbox.box_np_ops',
    points_cam2img=points_cam2img,
)
_mmdet3d_core_bbox.box_np_ops = _mmdet3d_core_bbox_box_np_ops


# ---------------------------------------------------------------------------
# 18. mmdet3d.datasets.pipelines  (removed in mmdet3d 1.x)
# ---------------------------------------------------------------------------

from mmdet3d.datasets import (                               # noqa: E402
    LoadAnnotations3D, LoadPointsFromFile,
    ObjectRangeFilter, ObjectNameFilter,
)


class _DefaultFormatBundle3D:
    """Minimal stub for DefaultFormatBundle3D (removed in mmdet3d 1.x).
    Subclasses (CustomDefaultFormatBundle3D) override __call__ and call super.
    """
    def __call__(self, results):
        import torch
        import numpy as np
        for key in ['points', 'gt_bboxes_3d', 'gt_labels_3d',
                    'gt_bboxes', 'gt_labels']:
            if key not in results:
                continue
            val = results[key]
            if hasattr(val, 'tensor'):
                results[key] = DataContainer(val)
            elif isinstance(val, np.ndarray):
                results[key] = DataContainer(torch.from_numpy(val))
            elif isinstance(val, torch.Tensor):
                results[key] = DataContainer(val)
        return results


_mmdet3d_datasets_transforms_3d = _make_module(
    'mmdet3d.datasets.pipelines.transforms_3d',
    ObjectRangeFilter=ObjectRangeFilter,
    ObjectNameFilter=ObjectNameFilter,
)
sys.modules['mmdet3d.datasets.pipelines.transforms_3d'] = _mmdet3d_datasets_transforms_3d

_mmdet3d_datasets_pipelines = _make_module(
    'mmdet3d.datasets.pipelines',
    DefaultFormatBundle3D=_DefaultFormatBundle3D,
    LoadAnnotations3D=LoadAnnotations3D,
    LoadPointsFromFile=LoadPointsFromFile,
    ObjectRangeFilter=ObjectRangeFilter,
    ObjectNameFilter=ObjectNameFilter,
    transforms_3d=_mmdet3d_datasets_transforms_3d,
)

import mmdet3d.datasets as _mmdet3d_datasets_mod2            # noqa: E402
_mmdet3d_datasets_mod2.pipelines = _mmdet3d_datasets_pipelines


# ---------------------------------------------------------------------------
# 18b. Register missing mmdet 1.x transformer layer types in mmengine MODELS
#      build_transformer_layer (mmcv 2.x) uses mmengine.MODELS internally.
#      DetrTransformerDecoderLayer → BaseTransformerLayer (same API, handles
#        legacy feedforward_channels / ffn_dropout via deprecated_args)
#      DetrTransformerEncoder       → TransformerLayerSequence
# ---------------------------------------------------------------------------

from mmcv.cnn.bricks.transformer import (                    # noqa: E402
    BaseTransformerLayer as _BTL,
    TransformerLayerSequence as _TLS,
)
from mmengine.registry import MODELS as _MMENGINE_MODELS_FINAL  # noqa: E402

if 'DetrTransformerDecoderLayer' not in _MMENGINE_MODELS_FINAL._module_dict:
    _MMENGINE_MODELS_FINAL.register_module(
        module=_BTL, name='DetrTransformerDecoderLayer', force=True)

if 'DetrTransformerEncoder' not in _MMENGINE_MODELS_FINAL._module_dict:
    _MMENGINE_MODELS_FINAL.register_module(
        module=_TLS, name='DetrTransformerEncoder', force=True)

# DeformableDetrTransformerDecoder compat shim (mmdet 1.x API):
# old API: TransformerLayerSequence subclass with transformerlayers=, return_intermediate=
# new mmdet 3.x API: completely different class, incompatible
import torch as _torch_tfd                                    # noqa: E402


class _DeformableDetrTransformerDecoderCompat(_TLS):
    """mmdet 1.x-compatible DeformableDetrTransformerDecoder.

    Inherits from TransformerLayerSequence (accepts transformerlayers/num_layers)
    and implements the reference-point-refining forward pass used in
    SegDeformableTransformer.
    """

    def __init__(self, *args, return_intermediate=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate

    def forward(self,
                query,
                *args,
                reference_points=None,
                valid_ratios=None,
                reg_branches=None,
                **kwargs):
        output = query
        intermediate = []
        intermediate_reference_points = []
        for lid, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                ref_pts = (
                    reference_points[:, :, None]
                    * _torch_tfd.cat(
                        [valid_ratios, valid_ratios], -1
                    )[:, None]
                )
            else:
                assert reference_points.shape[-1] == 2
                ref_pts = (
                    reference_points[:, :, None] * valid_ratios[:, None]
                )
            output = layer(
                output, *args, reference_points=ref_pts, **kwargs)
            if reg_branches is not None:
                # output is [seq, batch, embed]; reg_branches expect [batch, seq, embed]
                tmp = reg_branches[lid](output.permute(1, 0, 2))  # [batch, seq, dim]
                if reference_points.shape[-1] == 4:
                    new_ref = (
                        tmp + inverse_sigmoid(reference_points)
                    ).sigmoid()
                else:
                    new_ref = _torch_tfd.zeros_like(reference_points)
                    new_ref[..., :2] = (
                        tmp[..., :2] + inverse_sigmoid(reference_points[..., :2])
                    ).sigmoid()
                reference_points = new_ref.detach()
            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)
        if self.return_intermediate:
            return (_torch_tfd.stack(intermediate),
                    _torch_tfd.stack(intermediate_reference_points))
        return output, reference_points


_MMENGINE_MODELS_FINAL.register_module(
    module=_DeformableDetrTransformerDecoderCompat,
    name='DeformableDetrTransformerDecoder',
    force=True)

# Positional encodings (removed from mmengine::model in 3.x, live in mmdet.models.layers)
try:
    from mmdet.models.layers import (                        # noqa: E402
        LearnedPositionalEncoding as _LPE,
        SinePositionalEncoding as _SPE,
    )
    for _enc_name, _enc_cls in [('LearnedPositionalEncoding', _LPE),
                                 ('SinePositionalEncoding', _SPE)]:
        if _enc_name not in _MMENGINE_MODELS_FINAL._module_dict:
            _MMENGINE_MODELS_FINAL.register_module(
                module=_enc_cls, name=_enc_name, force=True)
except Exception:
    pass


# ---------------------------------------------------------------------------
# 18c. AnchorFreeHead compat shim
#      mmdet 3.x AnchorFreeHead has loss_by_feat / predict_by_feat as abstract
#      methods. Plugin classes (SegDETRHead, etc.) were written for mmdet 1.x
#      where AnchorFreeHead had no abstract methods.
#      Replace with a non-abstract BaseModule subclass.
# ---------------------------------------------------------------------------

from mmengine.model import BaseModule as _BaseModuleAFH      # noqa: E402


class _AnchorFreeHeadCompat(_BaseModuleAFH):
    """Non-abstract stub for mmdet 1.x AnchorFreeHead.

    Plugin code calls  super(AnchorFreeHead, self).__init__(init_cfg)
    which skips AnchorFreeHead's own __init__. We just need a non-abstract
    class with a compatible _load_from_state_dict stub.
    """

    def _load_from_state_dict(self, state_dict, prefix, local_metadata,
                               strict, missing_keys, unexpected_keys,
                               error_msgs):
        # mmdet 1.x version just calls super; replicate the same behaviour.
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)


import mmdet.models.dense_heads.anchor_free_head as _mmdet_afh_mod  # noqa: E402
_mmdet_afh_mod.AnchorFreeHead = _AnchorFreeHeadCompat

# Also patch the top-level mmdet.models.dense_heads namespace if already imported
import mmdet.models.dense_heads as _mmdet_dh2                # noqa: E402
_mmdet_dh2.AnchorFreeHead = _AnchorFreeHeadCompat
_mmdet_dh2.anchor_free_head = _mmdet_afh_mod


# ---------------------------------------------------------------------------
# 19. mmcv.cnn: bias_init_with_prob, xavier_init, constant_init, etc.
#     Moved from mmcv 1.x to mmengine.model.weight_init in mmcv 2.x
# ---------------------------------------------------------------------------

from mmengine.model.weight_init import (                     # noqa: E402
    bias_init_with_prob,
    xavier_init,
    constant_init,
    normal_init,
    uniform_init,
    kaiming_init,
)

import mmcv.cnn as _mmcv_cnn                                # noqa: E402
_mmcv_cnn.bias_init_with_prob = bias_init_with_prob
_mmcv_cnn.xavier_init = xavier_init
_mmcv_cnn.constant_init = constant_init
_mmcv_cnn.normal_init = normal_init
_mmcv_cnn.uniform_init = uniform_init
_mmcv_cnn.kaiming_init = kaiming_init


# ---------------------------------------------------------------------------
# 20. DETRHead compat shim (mmdet 1.x API → mmdet 3.x world)
#     BEVFormerTrackHead inherits from DETRHead and passes transformer=...
#     which the mmdet 3.x DETRHead no longer accepts.
# ---------------------------------------------------------------------------

from mmengine.model import BaseModule as _BaseModule          # noqa: E402


class _DETRHeadCompat(_BaseModule):
    """Minimal mmdet 1.x-compatible DETRHead base class.

    Sets up the same instance attributes that BEVFormerTrackHead (and other
    plugin heads derived from the 1.x DETRHead) rely on, then delegates layer
    construction to the subclass via _init_layers() / init_weights().
    """

    def __init__(self,
                 num_classes,
                 in_channels=None,
                 num_query=100,
                 num_reg_fcs=2,
                 transformer=None,
                 sync_cls_avg_factor=False,
                 positional_encoding=None,
                 loss_cls=dict(type='CrossEntropyLoss',
                               bg_cls_weight=0.1, use_sigmoid=False,
                               loss_weight=1.0, class_weight=1.0),
                 loss_bbox=dict(type='L1Loss', loss_weight=5.0),
                 loss_iou=dict(type='GIoULoss', loss_weight=2.0),
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 **kwargs):  # noqa: ARG002  # absorb extra 1.x kwargs
        super().__init__(init_cfg=init_cfg)
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.num_query = num_query
        self.num_reg_fcs = num_reg_fcs
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.fp16_enabled = False
        self.sync_cls_avg_factor = sync_cls_avg_factor

        # Build losses (use mmdet3d/mmdet registries via compat build)
        self.loss_cls = _COMBINED_MODELS.build(loss_cls) if isinstance(loss_cls, dict) else loss_cls
        self.loss_bbox = _COMBINED_MODELS.build(loss_bbox) if isinstance(loss_bbox, dict) else loss_bbox
        self.loss_iou = _COMBINED_MODELS.build(loss_iou) if isinstance(loss_iou, dict) else loss_iou

        if self.loss_cls.use_sigmoid:
            self.cls_out_channels = num_classes
        else:
            self.cls_out_channels = num_classes + 1

        # Build transformer via combined registry
        if transformer is not None:
            self.transformer = _COMBINED_MODELS.build(transformer)
            self.embed_dims = self.transformer.embed_dims

        # Build positional encoding (optional)
        if positional_encoding is not None:
            from mmcv.cnn.bricks.transformer import build_positional_encoding as _bpe
            self.positional_encoding = _bpe(positional_encoding)

        self._init_layers()

    def _init_layers(self):
        """Subclasses must override this to build their layers."""
        pass

    def init_weights(self):
        """Subclasses may override for custom weight init."""
        pass


# Patch mmdet.models.dense_heads.DETRHead with the compat version
import mmdet.models.dense_heads as _mmdet_dh                 # noqa: E402
_mmdet_dh.DETRHead = _DETRHeadCompat
# Also patch in the sys.modules path used by direct imports
_mmdet_dense_heads_mod = sys.modules.get('mmdet.models.dense_heads')
if _mmdet_dense_heads_mod is not None:
    _mmdet_dense_heads_mod.DETRHead = _DETRHeadCompat


# ---------------------------------------------------------------------------
# 21. HungarianAssigner compat shim (mmdet 1.x API → mmdet 3.x)
#     mmdet 1.x: HungarianAssigner(cls_cost=..., reg_cost=..., iou_cost=...)
#     mmdet 3.x: HungarianAssigner(match_costs=[...])
# ---------------------------------------------------------------------------

# Importing task_modules triggers TASK_UTILS registration of cost classes
import mmdet.models.task_modules  # noqa: E402

from mmdet.models.task_modules.assigners import (            # noqa: E402
    HungarianAssigner as _HungarianAssigner3x,
)
from mmdet.registry import TASK_UTILS as _MMDET_TASK_UTILS_REG  # noqa: E402


class _HungarianAssignerCompat(_HungarianAssigner3x):
    """Backward-compatible HungarianAssigner accepting mmdet 1.x kwargs.

    Old API: HungarianAssigner(
        cls_cost=dict(type='ClassificationCost', weight=1.),
        reg_cost=dict(type='BBoxL1Cost', weight=5.0),
        iou_cost=dict(type='IoUCost', iou_mode='giou', weight=2.0))
    New API: HungarianAssigner(match_costs=[...])
    """

    def __init__(self,
                 cls_cost=None,
                 reg_cost=None,
                 iou_cost=None,
                 match_costs=None):
        if match_costs is None:
            match_costs = []
            if cls_cost is not None:
                match_costs.append(cls_cost)
            if reg_cost is not None:
                match_costs.append(reg_cost)
            if iou_cost is not None:
                match_costs.append(iou_cost)
        super().__init__(match_costs=match_costs)


_MMDET_TASK_UTILS_REG.register_module(
    module=_HungarianAssignerCompat, name='HungarianAssigner', force=True)

