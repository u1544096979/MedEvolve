# 公共环境模块 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 提供独立远程医疗实例、原生工具、状态读取与患者轨迹整理。

**Architecture:** 管理请求到远程服务，业务工具到分配的实例地址。各环境适配负责数据结构和轨迹内容；上层使用统一接口。

**Tech Stack:** FastAPI、Docker Python SDK、httpx、OpenAI Agents SDK。

## Global Constraints

以 [主 specification](../design.md) 的第 5 节为接口约定，引用根目录 `contracts.py`。实施使用独立 worktree，不修改其他模块文件。使用替代响应进行本地验证；真实模型调用与服务器部署在配置和授权明确后执行。失败保留记录，不伪造成功输出。

## 文件归属与接口

创建 `server/app.py`、`server/docker_env.py`、`server/environments.json`、`environments/client.py`、`environments/api.py`、各环境适配文件，以及 `tests/environment/`。目录增加 __init__.py。公共导出集中在 api.py，签名为 specification 的五个环境接口。

环境配置选择实际适配模块，不定义继承树。适配文件分别为 `physicianbench.py`、`medagentbench.py`、`fhir_agentbench.py`。管理服务在远程安装，客户端在本机安装。

## Task 1：环境源码与运行能力

- [ ] 从各环境原始源码核对启动命令、数据位置、工具名称与参数、原生写入资源、状态查询方式，填写 server/environments.json 中对应镜像、服务端口、就绪路径和初始数据挂载。
- [ ] 每个环境独立实现读写与轨迹整理。没有实际源码或镜像的环境保持不可运行，配置选择时返回明确错误，不编造工具或共用另一个环境的实现。
- [ ] 验证本地替代服务可返回该环境原生响应；实际环境验证读取、原生支持的一次写入和状态查询。轨迹压缩方式在各适配内实现，保留时间、关键事实和来源，不由出题提示词代替。

## Task 2：远程实例生命周期

管理协议：POST /instances，输入 EnvironmentConfig 中 name 与 seed_data，输出 EnvironmentInstance；DELETE /instances/{instance_id}，输出 released: bool。server/environments.json 存储原生服务启动参数，本机无需知道容器内部结构。

- [ ] 实现独立容器端口、独立可写数据卷及服务器外部访问地址。就绪检查通过才返回 endpoint；失败清理本次资源。
- [ ] 客户端实现 create_environment、release_environment，将 HTTP 异常保留为环境错误。

```python
async with httpx.AsyncClient() as client:
    response = await client.post(f"{config.manager_url}/instances", json={
        "environment": config.name, "seed_data": config.seed_data})
    response.raise_for_status()
    instance = EnvironmentInstance.model_validate(response.json())
```

- [ ] 替代 Docker 接口验证端口和卷各自独立、启动失败清理、释放仅影响对应实例；执行 `pytest tests/environment/test_lifecycle.py -q`。
- [ ] 提交 `feat(environment): 实现远程实例生命周期`。

## Task 3：工具、状态和轨迹

- [ ] 实现 get_tools，使用 SDK function_tool 包装原生操作，闭包绑定 endpoint 与 patient_ids；工具参数描述按环境定义，拒绝读取范围之外患者。固定解题仅传单患者列表。
- [ ] 实现 read_state，依据任务内容及环境原生资源读取任务相关的写后状态，返回可序列化的原始字段；核验与完成评价负责解释这些字段。
- [ ] 实现 prepare_trajectory，完整分页读取单患者数据，按环境时间语义串联记录，去掉重复展示内容，返回 PatientTrajectory；每个 sources.source_id 均可定位原始资源。
- [ ] get_tools 在 decision_point 非空但时间限制未实现时直接抛出明确的 NotImplementedError。启用过滤后测试列表读取、按 ID 读取、关联读取三条路径，不能只过滤一种查询。

```python
@pytest.mark.asyncio
async def test_decision_point_is_not_silently_ignored(instance):
    with pytest.raises(NotImplementedError):
        await get_tools(instance, ["p1"], "2025-01-01T00:00:00+00:00")
```

- [ ] 用分页、重复记录和关键事实样例测试轨迹完整性与来源回查；运行 `pytest tests/environment -q`。实际创建两个同初始数据容器，写 A 后确认 B 未变，释放 A 后 B 仍可读取；提交 `feat(environment): 接入医疗工具与患者轨迹`。

## 交付验收

客户端能从本机访问原生 endpoint；工具范围与患者范围一致；轨迹保留来源；容器隔离通过。交付每个可运行环境的镜像／数据要求和验证结果，不将未验证的适配声明为完成。
