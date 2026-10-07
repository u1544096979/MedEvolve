# SingleShot 适配任务

SingleShot 是单次生成对比方法：每名评测患者一次模型调用，从全部五种题型中自主选题，同时生成最多两道任务、候选参考答案和逐条证据，不探索、不进化、不重试。

## 文档索引

- 方法定义与验收标准：[SPEC.md](SPEC.md)
- 实施步骤与文件职责：[PLAN.md](PLAN.md)

## 适配范围

1. 公共生成层支持按题型集合校验，其他方法保留 directions 的两个方向配额。
2. 方法独立维护生成函数与提示词，接收完整轨迹与五种题型，单次调用、最多两题。
3. 公共实验入口按 singleshot 分派自主选题参数；双环境配置复用对应 main 配置的环境、患者、模型角色与公共评测预算。
4. 在 PhysicianBench 与 MedAgentBench 上完成公共核验、独立解题与分阶段成本验收。

## 配置

- `configs/physicianbench-singleshot.json`
- `configs/medagentbench-singleshot.json`
