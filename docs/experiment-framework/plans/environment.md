# 公共环境实施计划

## 模块职责

创建与释放远程实例；包装原生工具；读取写后状态；整理单患者轨迹。

文件归属：environments/、server/。共享结构引用根目录 contracts.py。

## 接口

create_environment、get_tools、read_state、release_environment、prepare_trajectory。签名与数据结构见[主 specification](../design.md)第 5 节。

## 实施任务

核对原始环境源码并接入可用环境；实现远程 Docker 管理服务与本机客户端；每个实例独立可写存储。

实施使用独立 worktree；模型与环境配置来自实验配置。其他模块保持各自文件归属。

## 关键验收

创建两个实例，确认写入互不影响、释放互不影响；轨迹保留原始来源。

不为字段映射、JSON 往返和简单转发函数单独建立测试。公共闭环覆盖模块之间的调用关系。
