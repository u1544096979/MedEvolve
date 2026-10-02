# 主方法实施计划

## 模块职责

初始数据探索与环境探索；单患者一次出题；反馈分析与策略更新。

文件归属：methods/medevolve/。共享结构引用根目录 contracts.py。

## 接口

explore_data、explore_environment、generate_tasks、analyze_feedback、update_strategy。签名与数据结构见[主 specification](../design.md)第 5 节。

## 实施任务

两种探索分别产出抽象指导；生成接收轨迹、指导、策略与两个方向；反馈与更新各一次调用。

实施使用独立 worktree；模型与环境配置来自实验配置。其他模块保持各自文件归属。

## 关键验收

在完整闭环中确认初始探索一次、出题一次、下一轮使用更新策略。

不为字段映射、JSON 往返和简单转发函数单独建立测试。公共闭环覆盖模块之间的调用关系。
