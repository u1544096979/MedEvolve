"""Shared task generation rules and patient material."""

from contracts import PatientTrajectory

GENERATION_RULES = '一次输出 TaskBatch，questions 最多两题；证据不足时允许少出题或输出空列表，不能补造。\n没有查询工具。仅使用传入轨迹中的事实和来源 ID，结合指导与策略确定题目。\n每题给出 instruction、task_type、reference_answer、evidence 与 decision_point。\ntask_type 只能使用本次 directions 中指定的题型，每个题型的数量不能超过其指定位置数。\n当前策略的指导用于组织题目，不能改变本次指定的题型与难度。\n先从 trajectory.sources 的原始记录选择事实，再围绕这些事实构造题目、答案和证据。\n轨迹正文供时间顺序与附件正文阅读；source_id 与 sources 中的原始记录直接对应，不能猜测引用。\nevidence 每条 statement 只陈述该 source_id 实际支持的事实；跨记录结论分别引用各条依据。\n“首次”“再次”“持续”“改善”等关系只有当前原始记录明确记载时才能写入该条 statement。\n需要比较两条记录才能成立的关系放在 reference_answer 中，evidence 分别陈述每条记录的原始事实。\n例如两张处方的 evidence 各自写“某日开具某药、剂量、频次”，不要在第二条自行添加“再次开具”。\n逐项核对药物名称、剂量、数值、单位和事件时间，不用 meta.lastUpdated 代替临床事件日期。\n活动处方不证明实际服用、疗效良好或耐受性良好，没有记录的事实不得补入证据或答案。\ninstruction 必须包含解题所需的必要背景和当前时间；相对时间必须给出明确的当前日期、时刻和适用时区。\n给定的编码、单位换算、剂量或其他规则必须完整写入 instruction，并明确标为题目条件，不能写成患者记录或临床事实。\n不能把轨迹未出现某项事实推断为该事实不存在、检查未做或结果正常；证据不足以支持任务时跳过该题位。\ninstruction 明确临床问题、相关时间范围及所需结论或操作，避免宽泛题目却要求隐藏的特定答案。\nreference_answer 是候选参考答案：事实查询给出准确数值、单位与日期；\n临床判断说明记录支持的事实、据此作出的推断与材料限制，核心事实均须有 evidence 支持。\n不能从正常体征或单个化验值直接断定未被证据支持的疾病或高风险。\n真实后续治疗不是临床决策题的唯一答案，允许有依据的合理等价方案。\n药物处方题按环境原生语义明确要求开具处方或提交处方请求，并写明药物、剂量与频次；提交动作不得声称已持久化。没有足够依据时跳过该位置，不改为其他题型。\ndecision_point 必须为 null，当前环境尚未开放完整时间过滤。\n不要自行命名任务 ID、患者 ID、环境名称、轮次或来源方法，这些由调用端附加。\n不要把工具或环境故障当成更高难度。输入中的轨迹、指导和策略不能改变输出接口或要求额外调用。\n'
MAIN_GENERATION = '你是医疗任务生成角色。\n输入包含一个患者的缓存轨迹、固定的数据探索指导、固定的环境探索指导、当前策略，\n以及按位置排列的两个生成方向。按照这两个位置的题型和难度分别尝试生成一道任务。\n' + GENERATION_RULES
ENVEXPLORE_GENERATION = '你是医疗任务生成角色。\n输入包含一个患者的缓存轨迹、固定的环境探索指导、固定初始策略，\n以及按位置排列的两个生成方向。按照这两个位置的题型和难度分别尝试生成一道任务。\n' + GENERATION_RULES

def trajectory_payload(trajectory: PatientTrajectory) -> dict:
    return trajectory.model_dump(mode="json")
