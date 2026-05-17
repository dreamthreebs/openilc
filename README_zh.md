# OPENILC 中文说明

OPENILC 是一个用于 CMB 数据分析的 Internal Linear Combination 工具包。

现在项目分成三层：

- `openilc/`：可以安装的核心 Python 包，提供 `NILC` 和 `HILC`。
- 外部 CSV 配置：放在 `configs/` 目录。
- 教学辅助和脚本：放在 `tutorials/` 目录。

这样核心算法保持干净，配置和教学脚本也方便直接改。

## 安装

安装核心包：

```bash
pip install -e .
```

如果要运行教程里的 CAMB/PySM3 模拟数据生成，再安装可选依赖：

```bash
pip install -e ".[tutorial]"
```

## 快速使用

```python
import numpy as np
from openilc import NILC

maps = np.load("./test_data/sim_cfn.npy")

nilc = NILC.from_csv(
    "configs/bands.csv",
    "configs/needlets_default.csv",
    weights_name="./nilc_weight/w_map.npz",
    Sm_maps=maps,
    lmax=500,
    n_iter=1,
    weight_in_alm=False,
)

clean_map = nilc.run_nilc()
```

如果要做带 beam 的版本：

```python
nilc = NILC.from_csv(
    "configs/bands_beam.csv",
    "configs/needlets_beam.csv",
    Sm_maps=maps,
    lmax=1000,
    n_iter=1,
)
```

## 配置文件

配置文件故意放在包外面，方便教学和实验时修改。这里推荐使用 CSV，
因为 bands 和 needlets 本质上都是表格，横向比较更直观。

- `configs/bands.csv`：基础频段配置。
- `configs/bands_beam.csv`：考虑 beam 的频段配置。
- `configs/needlets_default.csv`：默认 needlet bins。
- `configs/needlets_beam.csv`：考虑 beam 的 needlet bins。

注意：

- `needlets` 最后一行的 `lmax` 应该和传给 `NILC` 的 `lmax` 一致。
- 相邻 needlet bin 的 `lmin`、`lpeak`、`lmax` 要连续、匹配。
- `nside` 要足够大，通常保持 `lmax < 2 * nside`。
- `lmax_alm` 会影响 NILC 实际用哪些频段。在 `calc_beta_for_scale` 中，
  如果某个频段的 `lmax_alm < beta_lmax`，这个频段会在该 needlet scale 被丢掉。

推荐的 CSV 用法：

```python
nilc = NILC.from_csv(
    "configs/bands_beam.csv",
    "configs/needlets_beam.csv",
    Sm_maps=maps,
    weights_name="./nilc_weight/w_alm.npz",
    lmax=1000,
    n_iter=1,
)
```

## 模拟数据

仓库不再保存大型二进制 `data/`。教程数据由 `tutorials/sim_data.py` 运行时生成：

- CMB power spectrum：用 CAMB 和 Planck 2018 类似参数生成。
- 前景：用 PySM3 生成 dust + synchrotron。
- 噪声：根据配置里的 `nstd` 生成 Gaussian pixel noise。

常用函数：

```python
from tutorials.sim_data import (
    estimate_lmax_from_beam,
    get_band_table,
    get_cmb_cls,
    get_foreground,
)
```

`estimate_lmax_from_beam(beam_arcmin, lmax, bl_floor=1e-4)` 保留下来是为了教学：
它展示 Gaussian beam transfer function 小到什么位置之后，deconvolution 会变得危险。
但 beam 教程真正使用的 `lmax_alm` 仍然来自 `configs/bands_beam.csv`，不会被这个估计值偷偷覆盖。

## 教程脚本

这些脚本是教程，不是 pytest 测试：

- `tutorials/tutorial_nilc.py`：基础 NILC，使用默认 CSV 配置。
- `tutorials/tutorial_nilc_beam.py`：考虑 beam 的 NILC，使用 beam CSV 配置。
- `tutorials/tutorial_ilc_bias.py`：ILC bias 检查。
- `tutorials/tutorial_cpr_ilc.py`：CPR/HILC 相关实验。

可以这样运行：

```bash
cd tutorials
python tutorial_nilc.py
python tutorial_nilc_beam.py
```

这些脚本会在本地生成被 git 忽略的目录，例如 `test_data/`、`test_data_beam/`、
`nilc_weight/`。

## 测试

真正的自动化测试放在 `tests/` 目录：

```bash
python -m pytest tests
```

教程脚本使用 `tutorial_*.py` 命名，所以不会被 pytest 当作单元测试收集。

## 导入方式

请使用包导入：

```python
from openilc import NILC, HILC
```

## 已知问题

- NILC 内存占用比较大。
- beta covariance matrix 的计算速度还比较慢。

## 高性能计算提示

如果 spherical harmonic transform 成为瓶颈，可以参考 healpy 的源码安装说明，
安装并配置 `cfitsio`、`healpix`，然后从源码重新编译 `healpy`。
