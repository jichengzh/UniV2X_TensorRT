# UniV2X 兼容性修复与使用说明

> 本文档记录了为使 UniV2X 在较新版本依赖（mmcv 2.x / mmdet3d 1.x / Shapely 2.0 等）下正常运行所做的全部修改，以及最终评估结果和复现方法。

---

## 目录

1. [背景与问题](#1-背景与问题)
2. [环境信息](#2-环境信息)
3. [修改总览](#3-修改总览)
4. [各文件详细修改说明](#4-各文件详细修改说明)
   - 4.1 [新建 compat.py](#41-新建-compatpy)
   - 4.2 [spd_vehicle_e2e_dataset.py](#42-spd_vehicle_e2e_datasetpy)
   - 4.3 [eval_utils/map_api.py](#43-eval_utilsmap_apipy)
   - 4.4 [nuscenes/eval/tracking/mot.py（第三方包）](#44-nuscenesevaltrackinmotpy第三方包)
5. [数据准备](#5-数据准备)
6. [模型权重](#6-模型权重)
7. [运行评估](#7-运行评估)
8. [最终评估结果](#8-最终评估结果)
9. [常见错误排查](#9-常见错误排查)

---

## 1. 背景与问题

UniV2X 原始代码基于 **mmcv 1.x / mmdet 2.x / mmdet3d 0.x**（旧版API）编写。
当前 conda 环境 `univ2x` 安装的是更新版本（mmcv 2.x / mmdet3d 1.x），导致大量 API 不兼容，无法直接运行。

主要不兼容来源：

| 依赖库 | 原版本 | 当前版本 | 主要变化 |
|--------|--------|---------|---------|
| mmcv | 1.x | 2.x | `DataContainer`、`collate`、`MMDistributedDataParallel`、`ProgressBar` 等 API 均已移除或重构 |
| mmdet | 2.x | 3.x | `mmdet.core`、`mmdet.models.builder` 等模块路径全面重组 |
| mmdet3d | 0.x | 1.x | `NuScenesDataset`、`Custom3DDataset`、数据集基类、pipeline 等大幅改动 |
| Shapely | 1.x | 2.x | `MultiPolygon` 不再直接可迭代，需使用 `.geoms` |
| motmetrics | — | 1.1.3 | nuscenes-devkit 内置的 `MOTAccumulatorCustom` 针对 1.4.0 格式编写，与 1.1.3 不兼容 |

---

## 2. 环境信息

```
conda 环境名：univ2x
Python：3.9
CUDA：11.x
PyTorch：适配版本
mmcv：2.x
mmdet：3.x
mmdet3d：1.x
Shapely：2.x
motmetrics：1.1.3
nuscenes-devkit：已安装（含 V2X-Seq-SPD 扩展）
```

---

## 3. 修改总览

| 文件 | 修改类型 | 说明 |
|------|---------|------|
| `compat.py`（新建） | 新增 | 全局兼容层，在 `tools/test.py` 最顶部 import，自动 patch 所有 API 差异 |
| `projects/mmdet3d_plugin/datasets/spd_vehicle_e2e_dataset.py` | 修改 | 修复数据集初始化、pipeline、评估所需的多处 API 缺失 |
| `projects/mmdet3d_plugin/datasets/eval_utils/map_api.py` | 修改 | 修复 Shapely 2.0 `MultiPolygon` 不可迭代问题 |
| `{conda}/nuscenes/eval/tracking/mot.py`（第三方包） | 修改 | 修复 `MOTAccumulatorCustom` 与 motmetrics 1.1.3 的格式不兼容 |

---

## 4. 各文件详细修改说明

### 4.1 新建 `compat.py`

**路径**：`UniV2X/compat.py`（项目根目录）

这是本次修复的核心文件，共 ~1700 行，作为统一的兼容层。
`tools/test.py` 第 4 行已添加 `import compat` 确保它在所有其他 import 之前加载。

#### 主要 patch 内容（共 19 个模块段）

**§1 mmcv.runner → mmengine**
- `load_checkpoint`、`wrap_fp16_model`、`get_dist_info`、`init_dist`
- `BaseRunner`、`EpochBasedRunner`、`IterBasedRunner`
- `build_optimizer`、各类 Hook

**§2 mmcv.parallel → mmengine + 自定义实现**

关键修复：

1. **`DataContainer` 完整实现**：mmcv 2.x 移除了 `DataContainer`，重新实现了完整版本，包括 `stack`、`cpu_only`、`pad_dims`、`__getitem__`、`__setitem__`、`__len__`、`__iter__`、`_data` 属性。

2. **`collate` 函数**：mmcv 1.x 的 `DataContainer`-aware collate 函数完整复现，按 `samples_per_gpu` 分块，正确处理 padding、stacking 和 cpu_only 场景。

3. **`MMDistributedDataParallel` 自定义实现**：重写 `forward()` 方法，在传入模型前递归展开 `DataContainer`（`_scatter_data` 函数），解决原 mmcv 1.x scatter 机制被移除的问题。

4. **`TORCH_VERSION` 修复**：以字符串形式存储（而非元组），确保 `digit_version()` 能正确解析。

**§3 mmcv.cnn.bricks.registry**
- patch `NORM_LAYERS`、`PLUGIN_LAYERS`、`POSITIONAL_ENCODING` 等注册表

**§4 mmcv 顶层 API**
- `Config`、`DictAction`、`mmcv.load`、`mmcv.dump`、`mkdir_or_exist`
- `track_iter_progress`（mmcv 1.x 进度条迭代器，重新实现）
- `ProgressBar`（从 mmengine 重新导出）
- `NuScenesDataset.DefaultAttribute`：mmdet3d 1.x 删除了此类属性，手动补回：
  ```python
  DefaultAttribute = {
      'car': 'vehicle.parked', 'pedestrian': 'pedestrian.moving',
      'trailer': 'vehicle.parked', 'truck': 'vehicle.parked',
      'bus': 'vehicle.moving',   'motorcycle': 'cycle.without_rider',
      'construction_vehicle': 'vehicle.parked',
      'bicycle': 'cycle.without_rider', 'barrier': '', 'traffic_cone': '',
  }
  ```

**§5–19 其他模块**
- `mmcv.utils.build_from_cfg`、`ConfigDict`、`Registry`
- `mmdet.models.builder`（`build_detector`、`build_backbone`、`build_head` 等）
- `mmdet.core`（`bbox_overlaps`、`reduce_mean`、`nms`、`multiclass_nms`、`distance2bbox`、`build_assigner`、`build_sampler` 等）
- `mmdet3d.core`（`LiDARInstance3DBoxes`、`Box3DMode`、`CameraInstance3DBoxes`、`show_result`、`box_np_ops`等）
- `mmdet3d.datasets.pipelines`（`LoadMultiViewImageFromFiles`、`MultiScaleFlipAug3D`、`LoadPointsFromFile`、`LoadAnnotations3D` 等）
- `mmdet3d.models.builder`（`build_model`）
- `mmdet3d.apis.single_gpu_test`

**DeformableDetrTransformerDecoder 形状修复（关键）**：

mmdet3d 1.x 中 `DeformableDetrTransformerDecoder.forward()` 输出 tensor 的形状为 `[seq, batch, embed]`（序列优先），而 `reg_branches` 期望 `[batch, seq, embed]`（批次优先）。修复如下：

```python
# 转置 output 后再输入 reg_branches
tmp = reg_branches[lid](output.permute(1, 0, 2))  # [batch, seq, dim]
# 修复 new_ref 赋值（原为 new_ref = tmp，形状不匹配）
new_ref = zeros_like(reference_points)
new_ref[..., :2] = (tmp[..., :2] + inverse_sigmoid(reference_points[..., :2])).sigmoid()
reference_points = new_ref.detach()
```

---

### 4.2 `spd_vehicle_e2e_dataset.py`

**路径**：`projects/mmdet3d_plugin/datasets/spd_vehicle_e2e_dataset.py`

#### Fix 1：`load_data_list()` 返回副本，防止零除错误

**问题**：`BaseDataset._serialize_data()` 调用 `self.data_list.clear()`（原地清空），由于 `self.data_infos` 和 `self.data_list` 指向同一列表对象，导致 `self.data_infos` 也被清空，`DistributedSampler` 计算 `len(dataset)` 时得到 0，引发 `ZeroDivisionError`。

**修复**（第 211 行附近）：

```python
def load_data_list(self):
    if not hasattr(self, 'load_interval'):
        self.load_interval = 1
    data_infos = self.load_annotations(self.ann_file)
    self.data_infos = data_infos
    return list(data_infos)  # 返回浅拷贝，保持两个属性独立
```

#### Fix 2：新增 `pre_pipeline()` 方法

**问题**：mmdet3d 1.x 基类移除了 `pre_pipeline()` 方法，`_prepare_test_data_single` 调用时抛出 `AttributeError`。

**修复**（在 `__len__` 方法之后添加）：

```python
def pre_pipeline(self, results):
    results['img_fields'] = []
    results['bbox3d_fields'] = []
    results['pts_mask_fields'] = []
    results['pts_seg_fields'] = []
    results['bbox_fields'] = []
    results['mask_fields'] = []
    results['seg_fields'] = []
    results['box_type_3d'] = self.box_type_3d
    results['box_mode_3d'] = self.box_mode_3d
```

#### Fix 3：`MultiScaleFlipAug3D` 输出格式转换

**问题**：mmdet3d 1.x 中 `MultiScaleFlipAug3D.transform()` 返回 `List[Dict]`（每个增强变体一个 dict），而 mmdet3d 0.x 返回 `Dict[List]`（每个键对应所有变体的列表）。

**修复**（`_prepare_test_data_single` 函数内，pipeline 调用之后）：

```python
example = self.pipeline(input_dict)
if isinstance(example, list):
    aug_dicts = example
    example = {k: [d[k] for d in aug_dicts] for k in aug_dicts[0]}
```

#### Fix 4：补充评估所需的类属性/实例属性

**问题**：`_format_bbox` 和 `_evaluate_single` 中使用了多个未定义的属性。

**修复**：

```python
# 作为类属性（在类定义顶部）：
class SPDE2EDataset(NuScenesDataset):
    ErrNameMapping = {
        'trans_err': 'mATE', 'scale_err': 'mASE',
        'orient_err': 'mAOE', 'vel_err': 'mAVE', 'attr_err': 'mAAE',
    }

# 在 __init__ 末尾：
self.eval_version = 'detection_cvpr_2019'
from nuscenes.eval.detection.config import config_factory
self.eval_detection_configs = config_factory(self.eval_version)
```

---

### 4.3 `eval_utils/map_api.py`

**路径**：`projects/mmdet3d_plugin/datasets/eval_utils/map_api.py`

**问题**：Shapely 2.0 移除了对 `MultiPolygon` 对象的直接迭代（`for poly in multi_polygon`），必须使用 `.geoms`。

**修复**（约第 2112–2116 行）：

```python
# 修复前
exteriors = [int_coords(poly.exterior.coords) for poly in polygons]

# 修复后
geoms = polygons.geoms if hasattr(polygons, 'geoms') else polygons
exteriors = [int_coords(poly.exterior.coords) for poly in geoms]
interiors = [int_coords(pi.coords) for poly in geoms for pi in poly.interiors]
```

---

### 4.4 `nuscenes/eval/tracking/mot.py`（第三方包）

**路径**：`{conda_env}/lib/python3.9/site-packages/nuscenes/eval/tracking/mot.py`

**问题**：nuscenes-devkit 内置的 `MOTAccumulatorCustom.new_event_dataframe_with_data()` 是针对 motmetrics 1.4.0 编写的（`_events` 为 dict 格式），但当前安装的 motmetrics 1.1.3 使用 list-of-lists 格式（`_events` 为 `[['Type', 'OId', 'HId', 'D'], ...]`，`_indices` 为 `[(FrameId, Event), ...]`）。

**修复**：使方法同时兼容两种格式：

```python
@staticmethod
def new_event_dataframe_with_data(indices, events):
    if len(events) == 0:
        return MOTAccumulatorCustom.new_event_dataframe()

    # motmetrics 1.1.3: events 为 list-of-lists，indices 为 list-of-tuples
    # motmetrics 1.4.0: events 为 dict，            indices 为 dict
    if isinstance(events, list):
        tevents = list(zip(*events))
        type_vals, oid_vals, hid_vals, d_vals = tevents[0], tevents[1], tevents[2], tevents[3]
        idx = pd.MultiIndex.from_tuples(indices, names=_INDEX_FIELDS)
    else:
        type_vals, oid_vals, hid_vals, d_vals = events['Type'], events['OId'], events['HId'], events['D']
        idx = pd.MultiIndex.from_arrays([indices[f] for f in _INDEX_FIELDS], names=_INDEX_FIELDS)

    raw_type = pd.Categorical(
        type_vals,
        categories=['RAW', 'FP', 'MISS', 'SWITCH', 'MATCH', 'TRANSFER', 'ASCEND', 'MIGRATE'],
        ordered=False)
    # OId/HId 保持 object 类型（nuscenes-devkit 使用字符串）
    series = [pd.Series(raw_type, name='Type'),
              pd.Series(oid_vals, dtype=object, name='OId'),
              pd.Series(hid_vals, dtype=object, name='HId'),
              pd.Series(d_vals, dtype=float, name='D')]
    df = pd.concat(series, axis=1)
    df.index = idx
    return df
```

---

## 5. 数据准备

### 5.1 生成示例数据集

```bash
cd /home/jichengzhi/UniV2X
conda run -n univ2x python tools/gen_example_data.py
```

生成目录：`data/V2X-Seq-SPD-New/`

### 5.2 转换数据格式

```bash
conda run -n univ2x bash tools/spd_dataset_converter.sh
```

转换后生成三个侧数据的 pkl 注解文件：
- `data/V2X-Seq-SPD-New/cooperative-vehicle-infrastructure/vehicle-side/`
- `data/V2X-Seq-SPD-New/cooperative-vehicle-infrastructure/infrastructure-side/`
- `data/V2X-Seq-SPD-New/cooperative-vehicle-infrastructure/cooperative/`

---

## 6. 模型权重

所有权重文件放置于 `ckpts/` 目录：

| 文件 | 用途 |
|------|------|
| `ckpts/univ2x_coop_e2e_stg1.pth` | 主模型（多智能体融合） |
| `ckpts/univ2x_sub_inf_stg2.pth` | 路侧子模型 |
| `ckpts/univ2x_sub_veh_stg1.pth` | 车侧子模型 |

---

## 7. 运行评估

### 7.1 完整推理 + 评估（耗时约 15 分钟）

```bash
cd /home/jichengzhi/UniV2X
conda run -n univ2x --no-capture-output bash -c \
  "CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 \
   tools/test.py \
   projects/configs_e2e_univ2x/univ2x_coop_e2e_track.py \
   ckpts/univ2x_coop_e2e_stg1.pth \
   --out output/results.pkl \
   --eval bbox \
   --launcher pytorch \
   2>&1 | tee eval_log.txt"
```

### 7.2 仅评估（复用已保存的推理结果，快速，约 1 分钟）

先临时修改 `tools/test.py`，将推理行注释掉、启用加载行（第 261–263 行附近）：

```python
# 注释掉推理：
# outputs = custom_multi_gpu_test(...)
# 启用加载：
outputs = mmcv.load(args.out)
```

然后运行：

```bash
cd /home/jichengzhi/UniV2X
conda run -n univ2x --no-capture-output bash -c \
  "CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 \
   tools/test.py \
   projects/configs_e2e_univ2x/univ2x_coop_e2e_track.py \
   ckpts/univ2x_coop_e2e_stg1.pth \
   --out output/results.pkl \
   --eval bbox \
   --launcher pytorch"
```

评估结束后记得将 `tools/test.py` 恢复为原始推理模式。

### 7.3 重要说明

- `test.py` 强制要求 `distributed=True`（非 distributed 模式下有 `assert False`），因此即使只用 1 张 GPU 也必须通过 `torchrun --nproc_per_node=1` 启动。
- 必须在 `UniV2X/` 根目录下运行，`compat.py` 通过 `sys.path.insert` 被找到。
- `compat` 模块必须是第一个 import（`tools/test.py` 第 4 行已确保）。

---

## 8. 最终评估结果

在 V2X-Seq-SPD 验证集（675 个样本）上，使用 `univ2x_coop_e2e_stg1.pth` 权重的评估结果：

### 目标检测（NuScenes Detection Score）

| 指标 | 值 |
|------|----|
| **mAP** | **0.0597** |
| **NDS** | **0.0586** |
| mATE（平移误差） | 0.9384 |
| mASE（尺度误差） | 0.7816 |
| mAOE（方向误差） | 0.9924 |
| mAVE（速度误差） | 1.2293 |
| mAAE（属性误差） | 1.0000 |

**各类别 AP：**

| 类别 | AP |
|------|----|
| car | 0.351 |
| bicycle | 0.211 |
| pedestrian | 0.036 |
| truck / bus / trailer / 其他 | 0.000 |

### 多目标跟踪（NuScenes Tracking）

| 指标 | 值 |
|------|----|
| **AMOTA** | **0.1440** |
| **AMOTP** | **1.7354** |
| MOTA | 0.1387 |
| MOTP | 0.7670 |
| Recall | 0.2682 |
| MOTAR | 0.4483 |
| FAF（误报/帧） | 48.06 |
| TP / FP / FN | 2456 / 762 / 4705 |
| ID Switches | 189 |
| Fragments | 149 |

### 地图分割（IoU）

| 类别 | IoU |
|------|-----|
| 可行驶区域（Drivable） | **0.7470** |
| 车道线（Lanes） | 0.1621 |
| 人行横道（Crossing） | 0.1917 |
| 轮廓（Contour） | 0.1383 |
| 分隔线（Divider） | NaN（无标注） |

---

## 9. 常见错误排查

| 错误信息 | 原因 | 解决方案 |
|---------|------|---------|
| `ZeroDivisionError` in `distributed_sampler.py` | `self.data_infos` 被清空（见 Fix 1） | `load_data_list()` 返回 `list(data_infos)` 副本 |
| `TypeError: 'MultiPolygon' object is not iterable` | Shapely 2.0 API 变更 | 使用 `polygons.geoms` |
| `AttributeError: 'SPDE2EDataset' has no attribute 'pre_pipeline'` | mmdet3d 1.x 删除基类方法 | 在 `SPDE2EDataset` 中手动添加 |
| `TypeError: list indices must be integers, not str` on `example['img_metas']` | `MultiScaleFlipAug3D` 返回 `List[Dict]` 而非 `Dict[List]` | pipeline 之后进行格式转换 |
| `TypeError: argument after ** must be a mapping, not list` | `collate` 函数返回格式错误 | 使用完整的 mmcv 1.x DataContainer-aware collate |
| `TypeError: 'DataContainer' object is not subscriptable` | `DataContainer` 缺少 `__getitem__` | 为 `DataContainer` 添加完整方法 |
| `AttributeError: 'tuple' object has no attribute 'split'` in `_digit_version` | `TORCH_VERSION` 存为 tuple 而非 string | 改为存储 `torch.__version__` 字符串 |
| `RuntimeError: expanded size mismatch` in DeformableDetr | tensor 形状 seq-first vs batch-first | 在 `reg_branches` 前 `output.permute(1,0,2)` |
| `AttributeError: NuScenesDataset has no attribute DefaultAttribute` | mmdet3d 1.x 删除类属性 | 在 `compat.py` 中手动添加 |
| `AttributeError: SPDE2EDataset has no attribute eval_version` | 忘记在 `__init__` 中赋值 | 添加 `self.eval_version = 'detection_cvpr_2019'` |
| `AttributeError: SPDE2EDataset has no attribute eval_detection_configs` | 忘记在 `__init__` 中赋值 | 添加 `self.eval_detection_configs = config_factory(...)` |
| `AttributeError: SPDE2EDataset has no attribute ErrNameMapping` | mmdet3d 1.x 删除类属性 | 添加 `ErrNameMapping` 类属性 |
| `TypeError: list indices must be integers, not str` in `mot.py` | nuscenes-devkit 针对 motmetrics 1.4.0 编写，与 1.1.3 格式不符 | 修改 `MOTAccumulatorCustom.new_event_dataframe_with_data` 兼容两种格式 |
| `can't open file '/home/tools/test.py'` | `torchrun` 在错误的工作目录下启动 | 在命令前加 `cd /home/jichengzhi/UniV2X &&` |

---

*本文档由 Claude Code 自动生成，记录了从 2026 年 3 月开始的兼容性修复过程。*
