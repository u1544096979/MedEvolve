# 模型运行实施计划

## 模块职责

定义共享数据结构；提供单次 LLM 调用和标准 SDK Agent。

文件归属：contracts.py、pyproject.toml、runtime/。共享结构引用根目录 contracts.py。

## 接口

call_llm、run_agent。签名与数据结构见[主 specification](../design.md)第 5 节。

## 实施任务

先建立共享结构与依赖；实现一次请求的结构化输出；使用 SDK Agent/Runner，记录轨迹、用量与预算状态。

实施使用独立 worktree；模型与环境配置来自实验配置。其他模块保持各自文件归属。

## 关键验收

通过可控响应运行一次工具调用和最终输出；验证预算能够终止运行。

不为字段映射、JSON 往返和简单转发函数单独建立测试。公共闭环覆盖模块之间的调用关系。
