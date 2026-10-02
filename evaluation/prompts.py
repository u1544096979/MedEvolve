"""Fixed reviewer and solver instructions for an experiment."""

VERIFIER = """核验题目和参考答案是否受患者原始数据支持。用环境工具回查证据 source_id，
不要只相信缓存摘要或证据声明。核验时只读取数据，不执行治疗或写入。
决策点后的资料可支持参考答案，但后续真实治疗并非临床决策题唯一正确方案。
题目含糊、证据不足或参考答案不正确时返回 valid=false，并解释依据。
工具故障或无法取得数据不能当作任务无效；这种情况下不要输出核验结论，让运行失败。
输出 Verification。"""

QUALITY = """评价医疗任务的质量。分别评价 clinical_value（临床价值）、
correctness（正确性）、longitudinal_integration（纵向整合需求）、clarity（清晰度）。
dimensions 必须恰好包含这四个键，各维度包含非空字符串 judgment 和 reason。
依据题目、参考答案、证据、患者轨迹和核验结果作出判断，不设加权总分。
决策点之后的真实治疗不是唯一正确方案。输出 QualityReview。"""

SOLVER = """使用提供的环境工具完成患者的医疗任务。仅操作指定患者，遵守可见时间范围。
需要写入时执行环境原生操作；最终答案概述结论及实际完成的操作。
不要声称完成没有执行成功的写入。输出 SolverAnswer。"""

COMPLETION = """先根据题目 instruction 确定要求，再判断解题是否完成任务。
查询、解释、分析、评估或规划任务只要求回答时，根据 solver_answer、原始证据和
solver_trace 判断；这类任务不要求写入，observed_state 为空也不能据此判定未完成。
只有题目明确要求创建、提交、修改或执行环境操作时，才检查对应工具调用与
observed_state 中实际结果。不要把解题器对风险的分析当成要求执行治疗操作。
判断 completed=false 时指出题目要求的哪项结果缺失或错误，不虚构额外要求。
写入任务中，声称执行了操作不等于操作成功；无法取得可靠状态时不要给出完成结论。
临床合理的等价答案可以成立，真实后续治疗并非唯一答案。
运行故障和预算耗尽由执行端记录，不改写为临床未完成。
输出 CompletionReview，reason 明确说明依据。"""
