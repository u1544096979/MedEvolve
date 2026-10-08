# 实验 1 图表数据接口

绘图入口为 `scripts/plot_experiment_figures.py`。图 2 的百分比分母为实际生成题数，图 3 展示核验通过任务的四维质量均值与预先计算的患者簇 95% bootstrap 区间。每个环境与生成模型组合独立生成两张图，方法顺序固定。

## 空白版式

```bash
.venv/bin/python scripts/plot_experiment_figures.py \
  --write-schema papers/figures/experiments/data-schema.json \
  --write-template papers/figures/experiments/data-template.json \
  --output-dir papers/figures/experiments
```

输出 `template-task-validity` 和 `template-task-quality` 的 PDF、SVG、PNG。数据为空时只显示坐标、方法、图例与缺失标记；不绘制结果条、均值点或区间。

## 正式数据

从 `data-template.json` 复制输入，在 `groups` 中按实验条件填写汇总。完整结构由 `data-schema.json` 定义。

| 层级 | 字段 | 含义 |
| --- | --- | --- |
| 根 | `version`、`groups` | 版本为 1；每个分组独立出图 |
| 分组 | `environment`、`generator`、`source_files`、`results` | 环境、生成模型、来源文件路径、六种方法汇总 |
| 方法 | `method`、`generated_tasks`、`generated_patients` | 方法配置键、实际生成题数、实际产题患者数 |
| 方法 | `outcomes` | 七种互斥分类的整数计数 |
| 方法 | `quality` | 四个质量维度的汇总，可分别缺失 |
| 维度 | `mean`、`ci_low`、`ci_high`、`n_tasks`、`n_patients` | 均值、区间两端、该维度有评分的题数与患者数 |

方法键为 `medevolve`、`taskcraft`、`workflow`、`singleshot`、`selfgen`、`envexplore`，每组恰好出现一次。显示名分别为 MedEvolve、TaskCraft、Workflow、SingleShot、SelfRefine、EnvExplore。正式观测值要求环境与生成模型均已填写，避免不同条件混画。

`outcomes` 包含 `invalid`、`evaluation_missing`、`runtime_error`、`level_1`、`level_2`、`level_3`、`level_4`。计数之和必须等于 `generated_tasks`。分类顺序为：核验未通过；核验通过且有难度层级；其余因核验或难度评价故障而缺失；其余评价缺失。后续解题或完成评审故障不覆盖已有难度层级。未产出题目的生成失败不计入该图分母，生成覆盖率另由论文表格报告。

`quality` 包含 `factual_accuracy`、`clinical_soundness`、`evidence_sufficiency`、`task_clarity`。均值来自逐题聚合分数，区间由患者簇重采样预先计算，脚本不重新抽样。计数使用各维度自身的评分覆盖。区间两端同时缺失时仍绘制已知均值，保留区间缺失状态；零评分样本用整个维度 `null`，不能给出均值 0。

未知值为 `null`，真实零计数为 `0`。空白模板的 `outcomes` 为整体 `null`；有分类汇总时必须同时提供七个计数。所有质量值范围为 1–5，区间包含均值，评分患者数不超过评分题数。不同裁判协议的数据应先分开审查并统一正式比较口径，再汇总输入。

```bash
.venv/bin/python scripts/plot_experiment_figures.py \
  --input outputs/paper-figures/experiment-1.json \
  --output-dir papers/figures/experiments
```

分组文件名前缀由环境与生成模型生成。每组同时保存可重新读取的 `*-data.json`，其中保留 `source_files` 与原始汇总值。SVG 保留文字，PDF 嵌入 TrueType 字体，PNG 为 300 DPI 预览。输出宽度固定 175 mm，有效性图高 88 mm，质量图高 126 mm；目标会议的最终尺寸需按其正式模板检查。

## 验证

```bash
.venv/bin/python -m pytest tests/experiment/test_figure_data.py -q
```

测试覆盖计数分区、缺失与零的区别、评分范围、区间完整性、样本量、方法和条件重复，以及空白输入不产生观测标记。绘图运行需要当前项目环境中的 Matplotlib 和 pydantic；当前版式以 Matplotlib 3.11.2 验证。

## 绘图规范来源

本工具的科学表达、缺失值、误差区间与矢量导出规范参考项目级 `scientific-visualization` 与 `matplotlib` skills。

Kassis, Timothy, Vinayak Agarwal, Yuhuan He, Darshil Patel, and Aubrey M. Brueckner. 2026. [Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents](https://arxiv.org/abs/2609.00065). arXiv:2609.00065.
