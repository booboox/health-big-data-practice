# 健康大数据实践课程资料与作业

按课程建立文件夹，整理课件、数据、实验 Notebook 与个人作业。

## 目录结构

```text
health-big-data-practice/
├── 02-data-types/                 # 数据类型与数据理解
├── 03-dimensionality-reduction/   # 高维数据与降维
├── 06-multiple-testing/           # 多重检验
├── 07-naive-bayes/                # 朴素贝叶斯
├── 08-linear-models/              # 线性模型、逻辑回归与 LDA
├── 09-svm/                        # 支持向量机
├── 10-ensemble/                   # 集成学习
├── SOURCE-MANIFEST.json           # 课程资料来源及文件校验值
└── README.md

每课目录/
├── slides/                       # 课件（或材料位置说明）
├── data/                         # 数据与来源说明
├── homework/                     # 个人作业与实验记录
├── notebook/ 或 notebooks/       # 实验 Notebook（按原包保留）
├── assets/                       # 图表、证据表等（如有）
└── 课程工作单、证据包及其他材料
```

## 课程索引

| 课次 | 主题 | 资料目录 | 个人作业 |
| --- | --- | --- | --- |
| 2 | 数据类型与数据理解 | [02-data-types](02-data-types/) | [homework](02-data-types/homework/)（待补充） |
| 3 | 高维数据与降维 | [03-dimensionality-reduction](03-dimensionality-reduction/) | [homework](03-dimensionality-reduction/homework/)（待补充） |
| 6 | 多重检验 | [06-multiple-testing](06-multiple-testing/) | [homework](06-multiple-testing/homework/)（待补充） |
| 7 | 朴素贝叶斯 | [07-naive-bayes](07-naive-bayes/) | [homework](07-naive-bayes/homework/)（待补充） |
| 8 | 线性模型、逻辑回归与 LDA | [08-linear-models](08-linear-models/) | [homework](08-linear-models/homework/)（已收集本地作业/实验文件） |
| 9 | 支持向量机（SVM） | [09-svm](09-svm/) | [homework](09-svm/homework/)（已收集本地作业/实验文件） |
| 10 | 集成学习 | [10-ensemble](10-ensemble/) | [homework](10-ensemble/homework/)（已收集本地作业/实验文件） |

## 整理说明

- 目前收到 7 个课程资料包；第 1、4、5 课暂未提供，不预设其主题。
- 课次采用压缩包内部目录或课件标识：例如外部 `5-multiple-test.zip` 内为第 6 课，`6.Naive-Bayes.zip` 内为第 7 课，`7. Linear-models (1).zip` 内为第 8 课，`9. Ensemble (1).zip` 内为第 10 课。
- 课件主要位于各课 `slides/`；第 3 课原包未提供独立 PPT，阅读资料位于 `readings/`。
- 个人作业中，第 8 课收录已有 Notebook、工作单和课堂笔记；第 9 课收录原工作单和标注“已完成”的版本；第 10 课收录 WDBC 工作单、演示文稿、实验代码与配图。其他课的工作单模板保留在课程材料中，未标记为已完成作业。
- 原课程资料的 `data/`、`assets/`、Notebook 和代码目录尽量保持原相对结构。第 2 课另在 `data/` 提供主数据副本；第 8 课个人 Notebook 的本机绝对数据路径已改为相对路径。
- 保留课程资料中的数据来源、许可说明和参考文献；未收录制作/审核过程目录、缓存及临时文件。

## 使用方法

进入相应课程目录查看 `README.md`。如需运行 Python 实验，按该课 `requirements.txt` 安装依赖，并从相应 Notebook 所在目录打开文件。个人作业统一放入该课 `homework/`。
