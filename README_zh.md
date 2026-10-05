# 测试时自适应中的回源迟滞

[English](README.md)

研究模型经过无标签测试时自适应路径后，首次返回干净、接近源分布的数据时的行为。项目区分归一化与配置效应、参数更新的增量影响，并结合方法稳定性分析与 CIFAR-C 外部验证。

## 研究范围

- 比较不同自适应路径与方法的回源表现。
- 区分配置效应与更新引起的变化。
- 依据固定的上游来源记录比较基线行为。
- 保留负结果和失败实验。

前瞻性机制筛查未支持所测试的机制。结论限定在记录的模型、数据集、协议与配置范围内。

## 仓库结构

| 位置 | 内容 |
| --- | --- |
| [configs/](configs/) | 冻结的实验批次、数据集来源与方法状态契约 |
| [results/analysis/](results/analysis/) | 衍生结果、逐行表格与产物清单 |
| [paper/](paper/) | 论文、参考文献与图表 |
| [docs/](docs/) | 实现、完整性、主张与审查记录 |
| [tests/](tests/) | 分析与运行时回归检查 |

仓库根目录包含实验执行器、监督器和分析脚本。原始数据集与模型检查点需根据清单中的来源另行获取。

## 环境与测试

记录中的分析使用 Python 3.13。在 PowerShell 中执行：

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements-lock.txt -r requirements-analysis-lock.txt -r requirements-dev-lock.txt
.\.venv\Scripts\python -m pytest -q
```

复用记录中的 CUDA 12.8 软件栈时，将 `requirements-lock.txt` 替换为 `requirements-cuda-lock.txt`。

测试不会启动正式实验。完整实验还需要引用的数据集、源模型检查点、计算资源及实验准入记录。

## 来源与验证

现代基线根据固定的上游参考独立重实现并开展行为核对，来源 commit、文件哈希和许可证见 [modern_baseline_upstreams.json](configs/modern_baseline_upstreams.json)。

论文当前为研究稿件。主张可追踪性与投稿就绪度分别记录在 [docs/](docs/) 中。

## 引用与许可证

引用信息见 [CITATION.cff](CITATION.cff)。原创代码与技术文档采用 [MIT](LICENSE)。论文、图表、表格和衍生科研结果遵循 [SCIENTIFIC_ARTIFACT_NOTICE.md](SCIENTIFIC_ARTIFACT_NOTICE.md)，第三方条款见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
