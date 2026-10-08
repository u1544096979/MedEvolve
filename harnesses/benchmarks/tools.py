"""官方 Benchmark 的实例级医疗操作与文件产物工具。"""
from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urlsplit

import httpx
from agents import function_tool
from environments.medagentbench import NativeTools
from harnesses.medical_bridge import ToolSessions


class BenchmarkNativeTools(NativeTools):
    """按 MedAgentBench 官方协议操作独立实例，包括患者检索题。"""

    def __init__(self, endpoint, functions):
        super().__init__(endpoint, ["benchmark"], functions)

    async def execute(self, action):
        text = action.replace("```tool_code", "").replace("```", "").strip()
        if text.startswith(("GET ", "POST ")):
            kind, target = text.split(" ", 1)
            target = target.split("\n", 1)[0].strip()
            parts = urlsplit(target)
            if parts.scheme or parts.netloc or target.startswith("/") or ".." in parts.path.split("/"):
                return {"kind":"error", "reply":"Use an instance-relative FHIR path"}
            url = self.endpoint + target
            request = {"action":text, "url":url, "payload":None}
            if kind == "POST":
                try:
                    payload = json.loads(text.split("\n", 1)[1])
                except (ValueError, IndexError):
                    return {"kind":"POST", "request":request, "reply":"Invalid POST request"}
                request["payload"] = payload
                return {"kind":"POST", "request":request, "reply":
                    "POST request accepted and executed successfully. Please call FINISH if you have got answers for all the questions and finished all the requested tasks"}
            request_url = url + ("&" if "?" in url else "?") + "_format=json"
            async with httpx.AsyncClient(timeout=30, trust_env=False, follow_redirects=False) as client:
                response = await client.get(request_url)
            if response.status_code >= 300:
                return {"kind":"GET", "request":request, "reply":f"Error in sending the GET request: HTTP {response.status_code}",
                    "failure_category":"environment_error"}
            return {"kind":"GET", "request":request, "data":response.text, "reply":
                f"Here is the response from the GET request:\n{response.text}. Please call FINISH if you have got answers for all the questions and finished all the requested tasks"}
        if text.startswith("FINISH(") and text.endswith(")"):
            try:
                answers = json.loads(text[7:-1])
                if not isinstance(answers, list):
                    raise ValueError("answers must be a list")
            except ValueError:
                return {"kind":"error", "reply":"Invalid FINISH request"}
            return {"kind":"FINISH", "answers":answers, "request":{"action":text}, "reply":"FINISH"}
        return {"kind":"error", "reply":"Invalid action"}


def output_tool(job_dir):
    output = (Path(job_dir) / "workspace/output").resolve()
    output.mkdir(parents=True, exist_ok=True)

    @function_tool(failure_error_function=None)
    async def write_file(file_path: str, content: str, mode: str = "w") -> dict:
        """Write a benchmark deliverable under /workspace/output; mode is w or a."""
        path = Path(file_path)
        prefix = Path("/workspace/output")
        if not path.is_relative_to(prefix) or mode not in {"w", "a"}:
            raise ValueError("write_file requires /workspace/output and mode w or a")
        target = (output / path.relative_to(prefix)).resolve()
        if not target.is_relative_to(output):
            raise ValueError("write_file path leaves /workspace/output")
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open(mode, encoding="utf-8") as stream:
            stream.write(content)
        return {"status":"ok", "path":file_path, "bytes_written":len(content.encode())}
    return write_file


class BenchmarkSessions(ToolSessions):
    """仅官方测试使用实例级工具；合成任务的患者访问规则独立保留。"""

    def __init__(self, settings, job_dir, **kwargs):
        super().__init__(**kwargs)
        self.settings, self.job_dir = settings, Path(job_dir)

    async def list(self, session_id, *, client_id=None):
        tools = await super().list(session_id, client_id=client_id)
        session = self.inspect(session_id, client_id=client_id)
        if session.native is not None:
            tools[0]["description"] += "\nOfficial FHIR functions:\n" + json.dumps(session.native.functions, ensure_ascii=False)
        return tools

    async def open(self, task, environment, **kwargs):
        # 通过环境代理选择官方工具，复用原有会话身份、预算和释放逻辑。
        original = self.environments
        owner = self

        class EnvironmentProxy:
            def __getattr__(self, name):
                return getattr(original, name)

            async def get_tools(self, instance, patient_ids, **options):
                if instance.environment == "MedAgentBench":
                    functions = json.loads((Path(owner.settings["source_dir"]) /
                        "data/medagentbench/funcs_v1.json").read_text(encoding="utf-8"))
                    return BenchmarkNativeTools(instance.endpoint, functions)
                tools = await original.get_tools(instance, patient_ids, **options)
                return [*tools, output_tool(owner.job_dir)]

            async def read_state(self, instance, task):
                # 官方评分器读取仍存活的实例；不执行合成任务的状态提取。
                return None

        self.environments = EnvironmentProxy()
        return await super().open(task, environment, **kwargs)
