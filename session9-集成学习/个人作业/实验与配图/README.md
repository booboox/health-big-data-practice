# 个人实验与证据文件

本目录恢复自已有的 WDBC 个人作业目录。工作单 Word 原件仍在上一级，不作修改。

- `experiment_d_ensembles.py`：单树、Bagging、随机森林及预测稳定性。
- `experiment_d3_importance.py`：特征重要性比较。
- `experiment_e12_boosting.py`：AdaBoost 与梯度提升。
- `experiment_e3_learning_curves.py`：学习率与训练轮数。
- `experiment_f_fair_comparison.py`：相同交叉验证划分下的模型比较。
- `tree_depth_comparison.py`：树深度比较。
- `plot_*.py`：结果与概念配图。
- `工作单配图/`：已有图表、结果表及实验元数据。
- `演示文稿/`：已有课堂演示与讲稿。

## 运行与来源

先按本课原课程包的 `10. Ensemble/requirements.txt` 安装 Python 依赖，再执行对应实验脚本。实验会将新结果写入本目录及 `工作单配图/`，可能更新已有结果文件。

本次将 8 个分析/绘图脚本的数据位置从原电脑 Downloads 绝对路径改为仓库内的 `10. Ensemble/data/student`，其余实验计算逻辑保持不变；修改文件如下：

- `experiment_d3_importance.py`
- `experiment_d_ensembles.py`
- `experiment_e12_boosting.py`
- `experiment_e3_learning_curves.py`
- `experiment_f_fair_comparison.py`
- `plot_bagging_oob.py`
- `plot_worksheet_a.py`
- `tree_depth_comparison.py`

本次未重跑完整实验；已有数字、配图与文件来自原个人实验，不声称为本次新运行结果。`build_completed_worksheet.py` 和 `qa_worksheet.py` 属于原文档制作/检查工具，可能仍依赖本机 Word、PDF 工具或路径，不是运行模型所必需的脚本。
