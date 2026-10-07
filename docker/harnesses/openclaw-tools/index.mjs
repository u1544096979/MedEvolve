// MedEvolve 医疗工具插件。
//
// 通过锁定版本的 api.registerTool() 注册两个原生工具，清单在
// openclaw.plugin.json 的 contracts.tools 中声明这两个名字（锁定版本要求运行时
// 注册的工具必须与 contracts.tools 一致，见 upstreams.lock.json 的证据）。
//
// 插件不实现任何 FHIR 逻辑：每次调用都转发到本容器内本次尝试的医疗工具桥会话，
// 会话标识由容器端 worker 写在 <OPENCLAW_STATE_DIR>/tool-session.json。

import { readFileSync } from "node:fs";

import { Type } from "typebox";
import { definePluginEntry } from "openclaw/plugin-sdk/plugin-entry";

const SESSION_FILE = "tool-session.json";
const CLIENT_HEADER = "x-medevolve-client";

function readSession() {
  const stateDir = process.env.OPENCLAW_STATE_DIR;
  if (!stateDir) {
    throw new Error("缺少 OPENCLAW_STATE_DIR，无法定位本次尝试的医疗工具会话");
  }
  let parsed;
  try {
    parsed = JSON.parse(readFileSync(`${stateDir}/${SESSION_FILE}`, "utf8"));
  } catch (error) {
    throw new Error(`本次尝试还没有可用的医疗工具会话：${error.message}`);
  }
  if (!parsed.bridge || !parsed.session || !parsed.client) {
    throw new Error("医疗工具会话文件不完整：需要 bridge、session 与 client");
  }
  return parsed;
}

async function callBridge(path, init) {
  const session = readSession();
  const response = await fetch(
    `${session.bridge.replace(/\/+$/, "")}${path}`,
    {
      ...init,
      headers: {
        "content-type": "application/json",
        [CLIENT_HEADER]: session.client,
        ...(init?.headers ?? {}),
      },
    },
  );
  const text = await response.text();
  if (!response.ok) {
    throw new Error(`医疗工具桥返回 ${response.status}：${text.slice(0, 500)}`);
  }
  try {
    return JSON.parse(text);
  } catch {
    return { raw: text };
  }
}

function textResult(value) {
  const text = typeof value === "string" ? value : JSON.stringify(value, null, 2);
  return { content: [{ type: "text", text }], details: value };
}

export default definePluginEntry({
  id: "medevolve-medical-tools",
  name: "MedEvolve Medical Tools",
  description: "把医疗工具桥的组内会话暴露为 OpenClaw 原生工具",
  register(api) {
    api.registerTool({
      name: "medical_tools_list",
      description: "列出本次尝试可用的医疗工具名称、说明与参数模式。",
      parameters: Type.Object({}),
      async execute() {
        const session = readSession();
        return textResult(
          await callBridge(`/sessions/${session.session}/tools`, { method: "GET" }),
        );
      },
    });

    api.registerTool({
      name: "medical_tool_call",
      description:
        "调用本次尝试的一个医疗工具。name 必须是 medical_tools_list 返回的工具名，arguments 必须是符合该工具参数模式的对象。",
      parameters: Type.Object({
        name: Type.String({ description: "medical_tools_list 返回的工具名" }),
        arguments: Type.Record(Type.String(), Type.Unknown(), {
          description: "该工具的参数对象",
        }),
      }),
      async execute(_id, params) {
        const session = readSession();
        return textResult(
          await callBridge(`/sessions/${session.session}/calls`, {
            method: "POST",
            body: JSON.stringify({
              name: params.name,
              arguments: params.arguments ?? {},
            }),
          }),
        );
      },
    });
  },
});
