# MedEvolve 的 OpenClaw 基座镜像。
#
# 构建上下文是仓库根目录：
#   docker build -f docker/harnesses/openclaw.Dockerfile \
#     -t medevolve-harness-openclaw:2026.9.2-3928bad9 .
#
# 只安装锁定版本的上游源码与依赖锁；构建过程不读取个人 OpenClaw 配置、技能、
# 记忆或登录缓存。医疗工具以独立插件（openclaw-tools/）注册，不修改上游源码，
# 因此没有 patches/ 差异文件。
#
# 锁定依据见 upstreams.lock.json：v2026.9.2 的 peeled 提交
# 3928bad9badfcb6c7d140530435e806fb8092190，依赖锁 pnpm-lock.yaml。

FROM node:22.22.3-bookworm

ARG OPENCLAW_REF=v2026.9.2
ARG OPENCLAW_COMMIT=3928bad9badfcb6c7d140530435e806fb8092190

RUN corepack enable && corepack prepare pnpm@12.1.0 --activate

# 只取锁定提交；ref 或提交对不上就构建失败，不使用浮动 main 或 latest。
WORKDIR /opt/openclaw
RUN git init . \
 && git remote add origin https://github.com/openclaw/openclaw \
 && git fetch --depth 1 origin "${OPENCLAW_REF}" \
 && git checkout --detach FETCH_HEAD \
 && test "$(git rev-parse HEAD)" = "${OPENCLAW_COMMIT}"

# 上游 LICENSE 原文随源码保留，许可证标注不做修改。
RUN pnpm install --frozen-lockfile
RUN pnpm build
RUN chmod +x /opt/openclaw/openclaw.mjs \
 && ln -s /opt/openclaw/openclaw.mjs /usr/local/bin/openclaw

# 医疗工具插件：只声明 contracts.tools，并把调用转发到本容器内的工具桥会话。
COPY docker/harnesses/openclaw-tools /opt/medevolve/openclaw-tools

# 插件从上游安装目录解析 typebox 与 openclaw/plugin-sdk；插件目录不在 /opt/openclaw
# 下，所以显式链接这两个包，而不是往插件里装第二份 SDK。
RUN mkdir -p /opt/medevolve/openclaw-tools/node_modules \
 && ln -s /opt/openclaw/node_modules/typebox /opt/medevolve/openclaw-tools/node_modules/typebox \
 && ln -s /opt/openclaw /opt/medevolve/openclaw-tools/node_modules/openclaw

# 容器端 worker 与它需要的 MedEvolve 模块；四个基座的依赖不合并进主环境。
COPY contracts.py /opt/medevolve/contracts.py
COPY harnesses/__init__.py harnesses/contracts.py harnesses/worker_protocol.py \
     harnesses/snapshots.py /opt/medevolve/harnesses/
COPY harnesses/workers /opt/medevolve/harnesses/workers
ENV PYTHONPATH=/opt/medevolve

# 容器不挂载宿主机 HOME、数据集或 Docker socket；状态只走 /state 与 /scratch。
# 显式保活命令与 HarnessIsolation 传入的命令一致，避免默认 CMD 立即退出。
WORKDIR /workspace
CMD ["sleep", "infinity"]
