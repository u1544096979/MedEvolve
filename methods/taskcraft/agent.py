"""源码 ToolCallingAgent 的执行控制移植（提交 1e405e0）。

源码路径：``taskcraft/oagents/agents.py`` 的 ``ToolCallingAgent._run``（1401-1487）、
``step``（1488-1623）、``provide_final_answer``（445-489）、``planning_step``（618-866）与
``reflection_step``（868-1010），记忆映射取自 ``oagents/memory.py``。
提示词逐字取自 ``oagents/prompts/w_think_reflect/toolcalling_agent.yaml``，即 ``SearchAgent``
实际使用的 ``prompts_type``；许可与作者归属见 LICENSE-TaskCraft。

适配只替换模型请求端与患者工具执行端：循环、步骤调度、记忆与计数口径保持源码行为。
模型请求改走公共单次调用入口，因此一次 action 的编码是 JSON 对象，而不是 SDK 工具调用字段；
函数工具协议与原生文本协议只在动作编码与终答解析处不同。每个真实请求、每次工具调用都进入
同一公共账本，规划与反思只增加模型轮数，不增加 action step。
"""

import json
import re

from agents.tool_context import ToolContext

from contracts import RunResult, SolverAnswer

from .environment import is_native
from .schemas import (
    AgentThought, AnswerList, NativeAction, SearchAction, TrajectoryScore,
)

__all__ = ["DEFAULT_MAX_ACTION_STEPS", "REFLECTION_INTERVAL", "run_search_agent", "step_kind"]

REFLECTION_INTERVAL = 5
DEFAULT_MAX_ACTION_STEPS = 20
FINAL_ANSWER_TOOL = "final_answer"
TRUNCATE_LIMIT = 512

# 提示词逐字取自 taskcraft/oagents/prompts/w_think_reflect/toolcalling_agent.yaml：
# SearchAgent 实际使用的 prompts_type。
AGENT_PROMPTS = {
    'system_prompt': 'You are an expert assistant who can solve any task using tool calls. You will be given a task to solve as best you can.\nTo do so, you have been given access to some tools.\n\nThe tool call you write is an action: after the tool is executed, you will get the result of the tool call as an "observation".\nThis Action/Observation can repeat N times, you should take several steps when needed.\n\nYou can use the result of the previous action as input for the next action.\nThe observation will always be a string: it can represent a file, like "image_1.jpg".\nThen you can use it as input for the next action. You can do it for instance as follows:\n\nObservation: "image_1.jpg"\n\nAction:\n{\n  "name": "image_transformer",\n  "arguments": {"image": "image_1.jpg"}\n}\n\nTo provide the final answer to the task, use an action blob with "name": "final_answer" tool. It is the only way to complete the task, else you will be stuck on a loop. Note: When calling the final_answer tool, simply pass in the answer you need to output. Do not generate additional parameters. For example, use it like this: final_answer(\'The content you want to output\'). Also, you can directly use the final_answer tool; there\'s no need to import it via import statements or other means. So your final output should look like this:\nAction:\n{\n  "name": "final_answer",\n  "arguments": "insert your final answer here"\n}\n\n\nHere are a few examples using notional tools:\n---\nTask: "Generate an image of the oldest person in this document."\n\nAction:\n{\n  "name": "document_qa",\n  "arguments": {"document": "document.pdf", "question": "Who is the oldest person mentioned?"}\n}\nObservation: "The oldest person in the document is John Doe, a 55 year old lumberjack living in Newfoundland."\n\nAction:\n{\n  "name": "image_generator",\n  "arguments": {"prompt": "A portrait of John Doe, a 55-year-old man living in Canada."}\n}\nObservation: "image.png"\n\nAction:\n{\n  "name": "final_answer",\n  "arguments": "image.png"\n}\n\n---\nTask: "What is the result of the following operation: 5 + 3 + 1294.678?"\n\nAction:\n{\n    "name": "python_interpreter",\n    "arguments": {"code": "5 + 3 + 1294.678"}\n}\nObservation: 1302.678\n\nAction:\n{\n  "name": "final_answer",\n  "arguments": "1302.678"\n}\n\n---\nTask: "Which city has the highest population , Guangzhou or Shanghai?"\n\nAction:\n{\n    "name": "search_agent",\n    "arguments": "Population Guangzhou"\n}\nObservation: [\'Guangzhou has a population of 15 million inhabitants as of 2021.\']\n\n\nAction:\n{\n    "name": "search_agent",\n    "arguments": "Population Shanghai"\n}\nObservation: \'26 million (2019)\'\n\nAction:\n{\n  "name": "final_answer",\n  "arguments": "Shanghai"\n}\n\nAbove example were using notional tools that might not exist for you. You only have access to these tools:\n{%- for tool in tools.values() %}\n- {{ tool.name }}: {{ tool.description }}\n    Takes inputs: {{tool.inputs}}\n    Returns an output of type: {{tool.output_type}}\n{%- endfor %}\n\n{%- if managed_agents and managed_agents.values() | list %}\nYou can also give tasks to team members.\nCalling a team member works the same as for calling a tool: simply, the only argument you can give in the call is \'task\', a long string explaining your task.\nGiven that this team member is a real human, you should be very verbose in your task.\nHere is a list of the team members that you can call:\n{%- for agent in managed_agents.values() %}\n- {{ agent.name }}: {{ agent.description }}\n{%- endfor %}\n{%- else %}\n{%- endif %}\n\nHere are the rules you should always follow to solve your task:\n1. ALWAYS provide a tool call, else you will fail.\n2. Always use the right arguments for the tools. Never use variable names as the action arguments, use the value instead.\n3. Call a tool only when needed: do not call the search_agent if you do not need information, try to solve the task yourself.\nIf no tool call is needed, use final_answer tool to return your answer.\n4. Never re-do a tool call that you previously did with the exact same parameters.\n\nPlease remember to aswer the question in language as {language}, otherwise the answer will be considered as invalid.\nNow Begin! If you solve the task correctly, you will receive a reward of $1,000,000.',
    'planning.initial_facts': 'Below I will present you a task.\n\nYou will now build a comprehensive preparatory survey of which facts we have at our disposal and which ones we still need.\nTo do so, you will have to read the task and identify things that must be discovered in order to successfully complete it.\nDon\'t make any assumptions. Provide a thorough reasoning before giving the facts as the answer. \n\nHere is how you will structure this answer:\n**Answer Structure**\n1) Facts given in the task: list here the specific facts given in the task that could help you (there might be nothing here).\n2) Facts to look up: list here any facts that we may need to look up. Also list where to find each of these, for instance a website, a file... - maybe the task contains some sources that you should re-use here.\n3) Facts to derive: list here anything that we want to derive from the above by logical reasoning, for instance computation or simulation.\n\n**Return Format**\n1) You MUST first return your thinking and reasoning process and second return your answer in the format: **<think>your_think_process</think>\\n<answer>your_answer</answer>**.\n2) For your answer, Keep in mind that "facts" will typically be specific names, dates, values, etc. Your answer should use the below headings and do not add anything else.:\n    ### 1. Facts given in the task\n    ### 2. Facts to look up\n    ### 3. Facts to derive\n\nHere is the task:\n```\n{{task}}\n```\nNow begin!',
    'planning.initial_plan': "You are a world expert at making efficient plans to solve any task using a set of carefully crafted tools.\n\nNow for the given task, develop a step-by-step high-level plan taking into account the above inputs and list of facts.\nThis plan should involve individual tasks based on the available tools, that if executed correctly will yield the correct answer.\n\nHere is your task:\n\n**Task**:\n```\n{{task}}\n```\n\n**You can leverage these tools**:\n{%- for tool in tools.values() %}\n- {{ tool.name }}: {{ tool.description }}\n    Takes inputs: {{tool.inputs}}\n    Returns an output of type: {{tool.output_type}}\n{%- endfor %}\n\n{%- if managed_agents and managed_agents.values() | list %}\n**You can also give tasks to team members.**\nCalling a team member works the same as for calling a tool: simply, the only argument you can give in the call is 'task', a long string explaining your task.\nGiven that this team member is a real human, you should be very verbose in your task.\nHere is a list of the team members that you can call:\n{%- for agent in managed_agents.values() %}\n- {{ agent.name }}: {{ agent.description }}\n{%- endfor %}\n{%- else %}\n{%- endif %}\n\n**List of facts that you know**:\n```\n{{answer_facts}}\n```\n\n**Return Requirements**:\n1. Do not skip steps, do not add any superfluous steps. Only write the high-level plan, DO NOT DETAIL INDIVIDUAL TOOL CALLS.\n2. If you need to use the search tool and search for more than three keywords, please determine the search priority and conduct multiple searches, with a maximum of two keywords each time.\n3. You MUST first return your thinking and reasoning process and second return your answer in the format: **<think>your_think_process</think>\\n<answer>your_answer</answer>**.\n\nNow begin! Write your plan below.",
    'reflection.process_rm': 'You are a highly skilled node evaluation agent for search algorithm. Your task is to assign a PRM score (0-10) to each candidate node, guiding the search towards the optimal path effectively.\n\n## Objective ##\nYou will receive a historical trajectory of executed nodes during a tree search. Each node includes:\n- `step_number` — Depth of the node.\n- `observations` — Observations recorded during this step.\n- `action_output` — Direct action output from the step.\n- `model_output` — The raw output of the model (LLM).\n- `error` — Any errors encountered (can be None).\n- `score` — Previously calculated score (can serve as a reference).\n\nYour goal is to estimate how promising the most recent node in this trajectory is for advancing towards solving the user\'s original task.  \nYou may also consider the previous `score` values as a reference for incremental improvement or regression but use your judgment for final evaluation.\n\n## Evaluation Guidelines ##\nEvaluate the node using these critical points:\n\n### 1. Progress Towards Goal (Highest Priority)\n- Analyze if the most recent step clearly moves closer to solving the user\'s original task.\n- Prefer nodes demonstrating tangible advancements or producing new, useful information.\n- Penalize nodes with irrelevant actions or minimal/no progress.\n\n### 2. Loop Detection (Critical Check!)\n- Carefully compare the most recent node with the historical trajectory:\n- **Real Loops**: If the new node repeats *the same* observations, action output, and model output **with no new information** or contextual changes, it likely indicates the search is stuck. Assign a very low PRM score (0-2).\n- **Benign Repetitions**: If the agent reuses similar tools or strategies with *different parameters* or yields additional results, this is not necessarily a loop. Do not penalize such cases severely unless no new value is added.\n\n### 3. Error and Stability\n- If `error` is not None, penalize based on severity:\n  - **Fatal/Blocking errors**: Score close to 0-1.\n  - **Significant errors**: Score 1-3.\n  - **Minor or recoverable issues**: May still score in moderate ranges (e.g., 3-5).\n- Nodes with unclear or unstable `model_output` should also receive moderate to low scores.\n\n### 4. Reference to Previous Score\n- Compare the new node’s performance with previous ones:\n  - If it shows clear improvement, the score can be higher than the previous node.\n  - If it shows regression or minimal additional value, the score can be lower.  \n- Typically, only drastically different performance (e.g. loops, severe errors) justifies large score deviations.\n\n## PRM Scoring Criteria (0-10) ##\n\n| Score | Criteria Description |\n|-------|----------------------|\n| 9-10  | Node clearly and significantly advances towards the goal, showing efficient search behavior at a reasonable depth. |\n| 7-8   | Good progress towards the goal; minor issues such as slight depth inefficiency. |\n| 5-6   | Moderate progress; node is helpful but may have limited clarity or efficiency. |\n| 3-4   | Poor or unclear progress; depth inefficiency or uncertain benefit. |\n| 1-2   | Negligible progress, signs of true loops, repetitive or confusing actions, or certain error states. |\n| 0     | Severe issues: explicit loops, significant errors, or complete irrelevance to the goal. |\n\n## Final Evaluation Output\nPlease provide your analysis rationale first, then give the final score judgment.You must provide your evaluation in valid JSON format with the following structure:\n```json\n{\n  "analysis": "Comprehensive assessment including reasoning, solution path evaluation, and if applicable, explanation of failure points",\n  "score": [integer between 0-10]\n}\n',
    'reflection.update_facts_pre_messages': 'You are an expert in adjusting fact-gathering strategies based on historical trajectory analysis. \nBelow you will find a task and a record of historical attempts to solve it, including the analysis and score from trajectory evaluation. \nYour task is to re-express and generate a list of facts containing:\n    ### 1. Facts given in the task\n    ### 2. Facts that we have learned\n    ### 3. Facts still to look up\n    ### 4. Facts still to derive\n\nPay special attention to:\n    1) Using the analysis and score from the reflection to judge the effectiveness of historical trajectories;\n    2) Prioritizing direction adjustment if the trajectory shows loops or inefficiency (score ≤ 4);\n    3) Consolidating existing fact-derivation logic if significant progress is shown (score ≥ 7);\n    4) Ensuring all fact organization is tightly aligned with the task goal, so each fact category serves the final answer.\nBased on the above requirements and combined with historical trajectory analysis, complete the update of the fact list.\n',
    'reflection.update_facts_post_messages': 'We previously constructed a fact list, but based on the latest historical trajectory analysis (including analysis and score), the fact list needs updating.\n\n** Return Format Requirements **:\n1)  You MUST first return your thinking and reasoning process and second return your answer in the format: **<think>your_think_process</think>\\n<answer>your_answer</answer>**.\n2) Please update your fact list based on the previous historical trajectory analysis, using the headings:\n    ### 1. Facts given in the task\n    ### 2. Facts that we have learned\n    ### 3. Facts still to look up\n    ### 4. Facts still to derive\n\n** Special Notes **:\n    1) If the analysis indicates repeated ineffective operations (e.g., loops with score ≤ 2), specify new search directions in "Facts still to look up";\n    2) If the score shows a fact category sufficiently supports derivation (e.g., score ≥ 6), transfer it from "still to look up" to "have learned";\n    3) All fact updates must reflect the guiding role of trajectory analysis, ensuring fact-gathering strategies align with the task-solving path.\n\nNow write your new list of facts below.\n',
    'reflection.update_plan_pre_messages': 'You are a world-class strategic planner specializing in adaptive task resolution. Your role is to develop a revised action plan by synthesizing three critical inputs:\n    1) The original task requirements,\n    2) Historical trajectory analysis (including PRM score and failure/success insights),\n    3) The latest updated facts list (including facts given, learned, to lookup, and to derive).\n\n** Planning Imperatives **\n    1) Trajectory-Driven Adjustments:\n      - If the PRM score ≤ 4 (indicating inefficiency or loops), prioritize radical course correction by: re-evaluating tool selection and parameters, and eliminating redundant steps identified in the analysis.\n      - If the PRM score ≥ 7 (indicating progress), build on successful strategies while refining minor inefficiencies.\n    2) Facts-Driven Prioritization:\n      - Use the "Facts Still to Look Up" section to identify missing information that requires urgent search actions,\n      - Leverage "Facts Still to Derive" to outline computational or logical steps needed to transform existing data into actionable insights.\n    3) Plan Structure Requirements:\n      - The plan must include high-level steps that align with available tools (listed below for reference),\n      - Avoid granular tool call details; focus on strategic milestones,\n      - Ensure each step directly addresses gaps identified in the trajectory analysis or facts list.\n\nFind below the record of what has been tried so far to solve it. Then you will be asked to make an updated plan to solve the task.\nIf the previous tries so far have met some success, you can make an updated plan based on these actions.\nIf you are stalled, you can make a completely new plan starting from scratch.\n',
    'reflection.update_plan_post_messages': 'You are now tasked with updating the action plan using the latest trajectory evaluation and facts update. The prior plan may have encountered:\n    1) Successes that can be scaled (if PRM score ≥ 7),\n    2) Loops or errors that require course correction (if PRM score ≤ 4),\n    3) Gaps in information identified in the "Facts Still to Look Up/Derive" sections.\n\n** You\'re still working towards solving this task **:\n```\n{{task}}\n```\n** Here is the historical trajectory analysis **:\n```\ntrajectory_score: {{trajectory_score}}, trajectory_analysis: {{trajectory_analysis}}\n```\n**Here is the up to date list of facts that you know**:\n```\n{{facts_update}}\n```\n\n** Key Adjustment Guidelines **:\n    1) Trajectory Analysis Integration:\n        - If the analysis highlights a "Real Loop" (score 0-2), explicitly replace redundant steps with new tool strategies or search priorities,\n        - For nodes with minor errors (score 3-5), adjust tool parameters or input data based on error descriptions.\n    2) Facts-Driven Plan Iteration:\n        - For "Facts Still to Look Up", insert search steps as early milestones, prioritizing high-impact keywords (max 2 per search),\n        - For "Facts Still to Derive", outline logical transformations (e.g., calculations, simulations) as intermediate steps.\n    3) Plan Optimization Rules:\n        - Reduce step depth if prior plans were inefficient (e.g., merge redundant search steps),\n        - Maintain tool call uniqueness (never repeat identical parameters per rule 4),\n        - Explicitly reference which trajectory insights drove each plan modification.\n\n**You can leverage these tools**:\n{%- for tool in tools.values() %}\n- {{ tool.name }}: {{ tool.description }}\n    Takes inputs: {{tool.inputs}}\n    Returns an output of type: {{tool.output_type}}\n{%- endfor %}\n\n{%- if managed_agents and managed_agents.values() | list %}\n**You can also give tasks to team members.**\nCalling a team member works the same as for calling a tool: simply, the only argument you can give in the call is \'task\'.\nGiven that this team member is a real human, you should be very verbose in your task, it should be a long string providing informations as detailed as necessary.\nHere is a list of the team members that you can call:\n{%- for agent in managed_agents.values() %}\n- {{ agent.name }}: {{ agent.description }}\n{%- endfor %}\n{%- else %}\n{%- endif %}\n\n**Return Requirements**:\n1. Do not skip steps, do not add any superfluous steps. Only write the high-level plan, DO NOT DETAIL INDIVIDUAL TOOL CALLS.\n2. You MUST first return your thinking and reasoning process and second return your answer in the format: **<think>your_think_process</think>\\n<answer>your_answer</answer>**.\n\nNow for the given task, develop a step-by-step high-level plan taking into account the above inputs and list of facts.\nThis plan should involve individual tasks based on the available tools, that if executed correctly will yield the correct answer.\nBeware that you have {remaining_steps} steps remaining.\n\nNow write your new plan below.\n',
    'final_answer.pre_messages': "An agent tried to answer a user query but it got stuck and failed to do so. You are tasked with providing an answer instead. Here is the agent's memory:",
    'final_answer.post_messages': 'Based on the above, please provide an answer to the following user task. \nYou MUST first return your thinking and reasoning process and second return your answer in the format: **<think>your_think_process</think>\\n<answer>your_answer</answer>**.\n\nHere is your task:\n{{task}}',
}

# ------------------------------------------------------------------ 控制函数
def step_kind(step: int, reflection_interval: int = REFLECTION_INTERVAL) -> str:
    """源码 _run 的步骤调度：第 1 步规划，之后每 reflection_interval 步反思，其余为 action。"""
    if step == 1:
        return "planning"
    if step % reflection_interval == 1:
        return "reflection"
    return "action"


# ------------------------------------------------------------------ 提示词渲染
def _tool_catalog(tools) -> str:
    """来源模板的 Jinja 工具清单循环：只渲染本次真实绑定的检索工具。"""
    if is_native(tools):
        api_base = tools.endpoint.rstrip("/")
        return "\n".join(
            "- " + json.dumps(item, ensure_ascii=False).replace("{api_base}", api_base)
            for item in tools.functions
        )
    lines = []
    for tool in tools:
        schema = getattr(tool, "params_json_schema", None)
        lines.append(
            f"- {tool.name}: {(getattr(tool, 'description', '') or '').strip()}\n"
            f"    Takes inputs: {json.dumps(schema, ensure_ascii=False) if schema else '{}'}\n"
            f"    Returns an output of type: str"
        )
    return "\n".join(lines)


_JINJA_BRANCH = re.compile(r"\{%-?\s*if managed_agents.*?\{%-?\s*endif\s*-?%\}", re.S)
_JINJA_TOOLS = re.compile(r"\{%-?\s*for tool in tools\.values\(\)\s*-?%\}.*?\{%-?\s*endfor\s*-?%\}",
                         re.S)
_JINJA_VARIABLE = re.compile(r"\{\{\s*(\w+)\s*\}\}")


def _populate(template: str, variables: dict) -> str:
    """按 SearchAgent 的实际取值渲染来源提示词。

    managed_agents 为空，相关分支按空文本处理；工具清单由 Jinja 循环改为预渲染文本；
    其余变量取 {{name}} 形式；源码里的单花括号占位符（如 {remaining_steps}）保持逐字。
    """
    text = _JINJA_BRANCH.sub("", template)
    text = _JINJA_TOOLS.sub(lambda match: variables.get("tools", ""), text)
    return _JINJA_VARIABLE.sub(
        lambda match: str(variables.get(match.group(1), match.group(0))), text)


def _truncate(content: str, max_length: int = TRUNCATE_LIMIT) -> str:
    """移植 oagents.utils.truncate_content：超长内容保留首尾。"""
    if len(content) <= max_length:
        return content
    return (content[:max_length // 2]
            + f"\n..._This content has been truncated to stay below {max_length} characters_...\n"
            + content[-max_length // 2:])


def _memory_messages(memory: list[dict]) -> str:
    """移植 memory.py 各步骤的 to_messages：TaskStep、PlanningStep、ReflectionStep、ActionStep。"""
    parts: list[str] = []
    for item in memory:
        kind = item["kind"]
        if kind == "task":
            parts.append(f"New task:\n{item['task']}")
        elif kind == "planning":
            parts.append(f"[FACTS RESULT LIST]:\n{str(item['facts']).strip()}")
            parts.append(f"[PLAN RESULT]:\n{str(item['plan']).strip()}")
        elif kind == "reflection":
            parts.append(
                f"[HISTORY TRAJECTORY ANALYSIS]:\n{str(item['analysis']).strip()} "
                f"\n [HISTORY TRAJECTORY SCORE]:\n{item['score']}")
            parts.append(f"[FACTS RESULT LIST]:\n{str(item['facts']).strip()}")
            parts.append(f"[PLAN RESULT]:\n{str(item['plan']).strip()}")
        elif kind == "action":
            if item.get("model_output"):
                parts.append(str(item["model_output"]))
            if item.get("tool_calls"):
                parts.append("Calling tools:\n"
                             + json.dumps(item["tool_calls"], ensure_ascii=False))
            if item.get("observations"):
                call_id = item["tool_calls"][0]["id"] if item.get("tool_calls") else ""
                parts.append(f"Call id: {call_id}\nObservation:\n{item['observations']}")
            if item.get("error"):
                parts.append(f"Error:\n{item['error']}\nNow let's retry: take care not to repeat "
                             "previous errors! If you have retried several times, try a completely "
                             "different approach.\n")
    return "\n\n".join(parts)


def _memory_string(memory: list[dict]) -> str:
    """移植 write_memory_to_string_messages：供反思的 process reward model 使用。"""
    lines = []
    for index, item in enumerate(memory, start=1):
        kind = item["kind"]
        if kind == "task":
            body = f"[TaskStep]\n  task: {item['task']}"
        elif kind == "planning":
            body = (f"[PlanningStep]\n  facts that we knows: {_truncate(str(item['facts']))}\n"
                    f"  current plan: {_truncate(str(item['plan']))}")
        elif kind == "reflection":
            body = (f"[ReflectionStep]\n  analysis of historic trajectory: "
                    f"{_truncate(str(item['analysis']))}\n  score of historic trajectory: "
                    f"{item['score']}\n  facts that we knows: {_truncate(str(item['facts']))}\n"
                    f"  current plan: {_truncate(str(item['plan']))}")
        else:
            body = (f"[ActionStep]\n  step_number: {item.get('step')}\n  observations: "
                    f"{_truncate(str(item.get('observations')))}\n  action_output: "
                    f"{_truncate(str(item.get('action_output')))}\n  model_output: "
                    f"{_truncate(str(item.get('model_output')))}\n  error: "
                    f"{_truncate(str(item.get('error')))}\n  score: 0.0")
        lines.append(f"===== Step {index} =====\n{body}")
    return "\n".join(lines)


def _calling_text(calls) -> str:
    """源码 step 的 model_output 文本：Calling tool: '<name>' with arguments: <arguments>。"""
    return "; ".join(f"Calling tool: '{call.name}' with arguments: {call.arguments}"
                     for call in calls)


def _finish_output(answers, output_type: type) -> dict:
    """原生协议 FINISH 的终答解析：答案数组映射 AnswerList/SolverAnswer，单对象映射声明的类型。"""
    if output_type is AnswerList:
        return {"answer_list": [str(item) for item in answers]}
    if output_type is SolverAnswer:
        return {"answer": json.dumps(list(answers), ensure_ascii=False)}
    if len(answers) != 1 or not isinstance(answers[0], dict):
        raise ValueError("FINISH must contain one object matching the requested output type")
    return output_type.model_validate(answers[0]).model_dump(mode="json")


def _tool_arguments(arguments) -> str:
    if isinstance(arguments, str):
        return arguments
    return json.dumps(arguments, ensure_ascii=False)


# 动作编码为 JSON 后的补充约束：源码用 <think>/<answer> 文本与 SDK 工具调用字段，
# 公共单次调用入口使用结构化输出，因此把同一动作语义写成 JSON 约定。
FUNCTION_ACTION_NOTE = """# 医疗适配的动作编码
1. 每次只返回一个 JSON 对象：{"tool_calls": [{"name": "工具名", "arguments": {…参数…}}], "final_answer": null}。
   一次可以给出多个工具调用，它们会按顺序执行；参数用具体值，不要用变量名。
2. 结束检索时把 tool_calls 置空，并把答案放进 final_answer，取值为符合下面 schema 的 JSON 对象。
3. final_answer 也可以作为一个工具调用出现：{"name": "final_answer", "arguments": {…答案…}}，出现即结束。
4. 除上面列出的病历工具外，只允许调用 final_answer；不要臆造工具名。
输出 schema：
{{output_schema}}"""

NATIVE_ACTION_NOTE = """# 医疗适配的动作编码
1. 每次只返回一个 JSON 对象：{"action": "<动作文本>"}，不要附加其他字段。
2. 动作文本只能是 GET 检索，或按本环境的 FINISH 约定收尾；不能发送 POST。
3. FINISH 的答案数组按下面 schema 的字段组织。
输出 schema：
{{output_schema}}"""


class _SearchRun:
    """一次检索 Agent 调用的状态：源码的 memory.steps、请求计数与 trace。"""

    def __init__(self, *, ledger, stage: str, instructions: str, task: str, tools,
                 output_type: type, max_action_steps: int):
        self.ledger = ledger
        self.stage = stage
        self.instructions = instructions
        self.task = task
        self.tools = tools
        self.output_type = output_type
        self.max_action_steps = max_action_steps
        self.native = is_native(tools)
        self.catalog = _tool_catalog(tools)
        self.answer = None          # 源码 self.answer：本方法不接收参考答案
        self.memory: list[dict] = [{"kind": "task", "task": task}]
        self.trace: list[dict] = []
        self.usage: dict = {"model_turns": 0, "tool_calls": 0,
                            "input_tokens": None, "output_tokens": None, "total_tokens": None}
        self.action_steps = 0
        self.final: dict | None = None
        self.stop: tuple[str, str] | None = None

    # -------------------------------------------------------------- 请求端
    async def request(self, name: str, *, kind: str, step: int, instructions: str,
                      input_text: str, output_type: type):
        run = await self.ledger.llm(f"{self.stage}/{name}", instructions=instructions,
                                    input_text=input_text, output_type=output_type)
        self._bill(run)
        if run is None:
            self.stop = ("budget_exhausted", "cumulative model budget exhausted")
            return None
        if run.status != "completed":
            self.stop = (run.status, run.error or run.status)
            return None
        self.trace.append({"type": "model_response", "kind": kind, "step": step,
                           "stage": f"{self.stage}/{name}", "output": run.output})
        return run

    def _bill(self, run) -> None:
        usage = (run.usage or {}) if run is not None else {}
        self.usage["model_turns"] += int(usage.get("model_turns") or 0)
        for field in ("input_tokens", "output_tokens", "total_tokens"):
            value = usage.get(field)
            if value is not None:
                self.usage[field] = (self.usage[field] or 0) + value

    def output_schema(self) -> str:
        return json.dumps(self.output_type.model_json_schema(), ensure_ascii=False)

    def action_instructions(self) -> str:
        note = NATIVE_ACTION_NOTE if self.native else FUNCTION_ACTION_NOTE
        return "\n\n".join((
            self.instructions,
            _populate(AGENT_PROMPTS["system_prompt"], {"tools": self.catalog}),
            _populate(note, {"output_schema": self.output_schema()}),
        ))

    # -------------------------------------------------------------- 规划与反思
    async def planning(self) -> None:
        facts_run = await self.request(
            "planning/facts", kind="planning", step=1, instructions="",
            input_text=_populate(AGENT_PROMPTS["planning.initial_facts"], {"task": self.task}),
            output_type=AgentThought)
        if facts_run is None:
            return
        facts = AgentThought.model_validate(facts_run.output)
        plan_run = await self.request(
            "planning/plan", kind="planning", step=1, instructions="",
            input_text=_populate(AGENT_PROMPTS["planning.initial_plan"],
                                 {"task": self.task, "answer_facts": facts.answer,
                                  "tools": self.catalog}),
            output_type=AgentThought)
        if plan_run is None:
            return
        plan = AgentThought.model_validate(plan_run.output)
        self.memory.append({"kind": "planning", "facts": facts.answer, "plan": plan.answer,
                            "facts_think": facts.think, "plan_think": plan.think})
        self.trace.append({"type": "planning", "step": 1, "facts": facts.answer,
                           "facts_think": facts.think, "plan": plan.answer,
                           "plan_think": plan.think, "requests": 2})

    async def reflection(self, step: int) -> None:
        # 源码 reflection_step：answer = str(self.answer)，本方法没有参考答案，文本与源码一致。
        prm_input = f"standard_answer:{self.answer}\n\n\n" + _memory_string(self.memory)
        score_run = await self.request(
            "reflection/score", kind="reflection", step=step,
            instructions=AGENT_PROMPTS["reflection.process_rm"], input_text=prm_input,
            output_type=TrajectoryScore)
        if score_run is None:
            return
        score = TrajectoryScore.model_validate(score_run.output)

        history = _memory_messages(self.memory)
        facts_run = await self.request(
            "reflection/facts", kind="reflection", step=step,
            instructions=AGENT_PROMPTS["reflection.update_facts_pre_messages"],
            input_text=(history + "\n\n"
                        + AGENT_PROMPTS["reflection.update_facts_post_messages"]),
            output_type=AgentThought)
        if facts_run is None:
            return
        facts = AgentThought.model_validate(facts_run.output)
        plan_run = await self.request(
            "reflection/plan", kind="reflection", step=step,
            instructions=AGENT_PROMPTS["reflection.update_plan_pre_messages"],
            input_text=(history + "\n\n" + _populate(
                AGENT_PROMPTS["reflection.update_plan_post_messages"],
                {"task": self.task, "tools": self.catalog,
                 "trajectory_score": score.score, "trajectory_analysis": score.analysis,
                 "facts_update": facts.answer,
                 "remaining_steps": self.max_action_steps - step,
                 "standard_answer": self.answer})),
            output_type=AgentThought)
        if plan_run is None:
            return
        plan = AgentThought.model_validate(plan_run.output)
        self.memory.append({"kind": "reflection", "analysis": score.analysis,
                            "score": score.score, "facts": facts.answer,
                            "facts_think": facts.think, "plan": plan.answer,
                            "plan_think": plan.think})
        self.trace.append({"type": "reflection", "step": step, "analysis": score.analysis,
                           "score": score.score, "facts": facts.answer, "plan": plan.answer,
                           "requests": 3})

    # -------------------------------------------------------------- action
    async def action(self, step: int) -> None:
        run = await self.request(
            "action", kind="action", step=step, instructions=self.action_instructions(),
            input_text=_memory_messages(self.memory),
            output_type=NativeAction if self.native else SearchAction)
        if run is None:
            return
        self.action_steps += 1
        try:
            if self.native:
                await self._native_action(step, run)
            else:
                await self._function_action(step, run)
        except Exception as exc:
            self.stop = ("failed", f"invalid action response: {type(exc).__name__}: {exc}")

    def _finish(self, value, *, kind: str, step: int, entry: dict | None = None) -> bool:
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except ValueError:
                pass
        try:
            self.final = self.output_type.model_validate(value).model_dump(mode="json")
        except Exception as exc:
            self.stop = ("failed", f"invalid final answer: {type(exc).__name__}: {exc}")
            return False
        if entry is not None:
            entry["action_output"] = self.final
        self.trace.append({"type": "final_answer", "kind": kind, "step": step,
                           "answer": self.final})
        return True

    async def _function_action(self, step: int, run) -> None:
        action = SearchAction.model_validate(run.output)
        entry = {"type": "action", "step": step, "model_output": _calling_text(action.tool_calls),
                 "tool_calls": [], "observations": None}
        self.trace.append(entry)
        observations = []
        finished = False
        for call in action.tool_calls:
            call_id = f"call-{len(entry['tool_calls']) + 1}"
            entry["tool_calls"].append({"name": call.name, "arguments": call.arguments,
                                        "id": call_id})
            if call.name == FINAL_ANSWER_TOOL:
                finished = self._finish(call.arguments, kind="action", step=step, entry=entry)
                break
            observation = await self._call_function(step, call_id, call)
            if observation is None:
                break
            observations.append(observation)
        if not finished and self.stop is None and not action.tool_calls \
                and action.final_answer is not None:
            # 源码在 tool_calls 非空时只执行工具调用并继续循环，不解析消息内容，
            # 因此 final_answer 字段只在同一次响应不含工具调用时才作为终答。
            finished = self._finish(action.final_answer, kind="action", step=step, entry=entry)
        if not finished and self.stop is None and not observations:
            # 源码 step：JSON 动作里既没有工具名也没有终答，视为生成失败。
            self.stop = ("failed", "action response carries no tool call")
        if observations:
            entry["observations"] = "\n".join(observations)
        self.memory.append({"kind": "action", "step": step, "model_output": entry["model_output"],
                            "tool_calls": entry["tool_calls"],
                            "observations": entry["observations"]})

    async def _call_function(self, step: int, call_id: str, call) -> str | None:
        if self.ledger.remaining_tools() == 0:
            self.stop = ("budget_exhausted", "cumulative tool call budget exhausted")
            return None
        arguments = _tool_arguments(call.arguments)
        available = {getattr(tool, "name", ""): tool for tool in self.tools}
        tool = available.get(call.name)
        if tool is None:
            observation = (f"Unknown tool {call.name}, should be instead one of "
                           f"{list(available)}.")
        else:
            context = ToolContext(context=None, tool_name=call.name, tool_call_id=call_id,
                                  tool_arguments=arguments)
            try:
                observation = await tool.on_invoke_tool(context, arguments)
            except Exception as exc:
                observation = f"{type(exc).__name__}: {exc}"
        observation = str(observation)
        self.usage["tool_calls"] += 1
        self.ledger.tool(f"{self.stage}/tool", name=call.name, arguments=arguments,
                         output=observation)
        self.trace.append({"type": "tool_start", "step": step, "name": call.name,
                           "call_id": call_id, "arguments": arguments})
        self.trace.append({"type": "tool_output", "step": step, "name": call.name,
                           "call_id": call_id, "output": observation})
        return observation

    async def _native_action(self, step: int, run) -> None:
        action = NativeAction.model_validate(run.output).action.strip()
        entry = {"type": "action", "step": step, "model_output": action,
                 "tool_calls": [], "observations": None}
        self.trace.append(entry)
        if action.startswith("FINISH(") and action.endswith(")"):
            result = await self._call_native(action, count=False)
            if result is None:
                return
            if result.get("kind") != "FINISH":
                self.stop = ("failed", "Invalid FINISH action: "
                             f"{result.get('reply', 'invalid response')}")
                return
            try:
                self._finish(_finish_output(result.get("answers", []), self.output_type),
                             kind="action", step=step, entry=entry)
            except Exception as exc:
                self.stop = ("failed", f"invalid FINISH output: {type(exc).__name__}: {exc}")
            self.memory.append({"kind": "action", "step": step, "model_output": action,
                                "tool_calls": [], "observations": None})
            return
        result = await self._call_native(action, count=True)
        if result is None:
            return
        observation = str(result.get("reply", ""))
        if result.get("kind") == "error":
            entry["error"] = observation
            self.stop = ("failed", f"Invalid native action: {observation}")
        else:
            entry["observations"] = observation
        self.memory.append({"kind": "action", "step": step, "model_output": action,
                            "tool_calls": [], "observations": entry["observations"]})

    async def _call_native(self, action: str, *, count: bool):
        if count and self.ledger.remaining_tools() == 0:
            self.stop = ("budget_exhausted", "cumulative tool call budget exhausted")
            return None
        try:
            result = await self.tools.execute(action)
        except Exception as exc:
            result = {"kind": "error", "reply": f"{type(exc).__name__}: {exc}"}
        if count:
            self.usage["tool_calls"] += 1
            self.ledger.tool(f"{self.stage}/tool", name="native_action", arguments=action,
                             output=result.get("reply", ""))
            self.trace.append({"type": "tool_start", "name": "native_action", "arguments": action})
            self.trace.append({"type": "tool_output", "name": "native_action", "output": result})
        return result

    # -------------------------------------------------------------- 强制终答
    async def provide_final_answer(self) -> None:
        step = self.max_action_steps + 1
        post = _populate(AGENT_PROMPTS["final_answer.post_messages"], {"task": self.task})
        run = await self.request(
            "forced_final", kind="forced_final", step=step,
            instructions=AGENT_PROMPTS["final_answer.pre_messages"],
            input_text=f"{_memory_messages(self.memory)}\n\n{post}",
            output_type=AgentThought)
        if run is None:
            return
        thought = AgentThought.model_validate(run.output)
        # 源码 provide_final_answer 返回 <think>/<answer> 文本，SearchAgent.forward 用
        # safe_json_loads 解析 answer；适配在同一位置把 answer 解析为声明的输出结构。
        self._finish(thought.answer, kind="forced_final", step=step)

    def result(self) -> RunResult:
        if self.final is not None:
            status, error = "completed", None
        elif self.stop is not None:
            status, error = self.stop
        else:
            status, error = "failed", "search agent produced no answer"
        usage = {**self.usage, "action_steps": self.action_steps,
                 "step_number": self.action_steps + 1,
                 "max_action_steps": self.max_action_steps}
        return RunResult(output=self.final, trace=self.trace, usage=usage, status=status,
                         error=error)


async def run_search_agent(*, ledger, stage: str, instructions: str, input_text: str, tools,
                           output_type: type,
                           max_action_steps: int = DEFAULT_MAX_ACTION_STEPS) -> RunResult:
    """按源码 ToolCallingAgent._run 执行一次检索：规划、反思、action、强制终答。

    模型请求走 Ledger 的单次调用入口并逐次记账；达到 action 上限后仍发起源码的
    provide_final_answer 请求，累计预算不足时该请求被公共账本截断。
    """
    run = _SearchRun(ledger=ledger, stage=stage, instructions=instructions, task=input_text,
                     tools=tools, output_type=output_type, max_action_steps=max_action_steps)
    step = 1
    while step <= max_action_steps and run.final is None and run.stop is None:
        kind = step_kind(step, REFLECTION_INTERVAL)
        if kind == "planning":
            await run.planning()
        elif kind == "reflection":
            await run.reflection(step)
        if run.final is not None or run.stop is not None:
            break
        await run.action(step)
        step += 1
    if run.final is None and run.stop is None:
        await run.provide_final_answer()
    return run.result()
