# MedEvolve 英文论文

## 稿件与模板

英文稿沿用已确认中文稿的章节及研究范围，采用用户提供的 ACL 模板压缩包内原版 `acl.sty` 与 `acl_natbib.bst`，preprint 阅读模式，保留页码。源文件、引用库、图片和 PDF 位于 `papers/english/`，可独立编译。

## 正文骨架

1. Introduction and Related Work
2. Problem Formulation and Method
3. Experimental Setup
4. Results and Analysis：生成主比较、四轮策略自进化、四组组件消融、下游经验学习。
5. Discussion and Conclusion
6. Limitations、References、Appendices A–G

## 内容约束

保留患者划分、评价分母、冻结初始采样配比及两轮已有运行与四轮拟开展实验的区别。正式结果为空，摘要不宣称尚无结果支持的实验收益。保留五张图、九张表及其结果待填状态。作者与参考文献元数据不补造；作者核验状态沿用原证据记录。

## 主文篇幅与展示

主文按 NAACL 长论文的八页上限组织：摘要与引言、方法、实验设置、四块结果、结论。方法概览、任务有效性和自进化曲线在主文；指导修订示例及四维质量图在附录。主实验产出与质量合并成一张表，保留自进化、消融及下游经验学习的关键比较。患者分批、评分尺度、证据结构及 Harness 更新机制放入附录。模板字体、边距和行距采用原版样式。
