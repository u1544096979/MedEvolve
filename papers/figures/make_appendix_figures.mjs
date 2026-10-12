// Generates appendix prompt figures (SVG). Render to PDF/PNG with render_appendix_figures.sh.
import { writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const OUT = dirname(fileURLToPath(import.meta.url));

const C = {
  ink: "#17324D", muted: "#52677B", teal: "#087F83", tealDark: "#066568",
  orange: "#C45B25", edge: "#CEDCE5", pale: "#EFF7F8", warm: "#FFF5ED",
  white: "#FFFFFF", bandTeal: "#D5EEEE", bandOrange: "#F8E8DC", gray: "#F5F7F9",
  grayEdge: "#D0D8E0", red: "#B3261E", redPale: "#FDECEA", green: "#2E7D4F", greenPale: "#E8F4EC",
};
const SANS = "'Noto Sans SC','Arimo','Liberation Sans',Arial,sans-serif";
const MONO = "'Roboto Mono','DejaVu Sans Mono',monospace";

const esc = (s) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
const cw = (ch, fs) => (/[⺀-鿿＀-￯　-〿—…‘-”]/.test(ch) ? fs : fs * 0.56);
const tw = (s, fs) => [...s].reduce((a, ch) => a + cw(ch, fs), 0);

function wrap(text, fs, maxW) {
  const units = text.match(/[A-Za-z0-9_.=`\-+<>≤≥\[\]{}()\/:;,]+|\s+|[\s\S]/gu) || [];
  const noStart = new Set("，。、；：？！）」』》】,.;:!?)".split(""));
  const lines = [];
  let cur = "", w = 0;
  for (const u of units) {
    const uw = tw(u, fs);
    if (w + uw > maxW && cur && !(u.length === 1 && noStart.has(u))) {
      lines.push(cur.trimEnd()); cur = ""; w = 0;
      if (/^\s+$/.test(u)) continue;
    }
    cur += u; w += uw;
  }
  if (cur.trim()) lines.push(cur.trimEnd());
  return lines;
}

// inline markup: `code` -> mono, plain otherwise. Returns tspans.
function runs(line, fs, fill, weight = 400) {
  return line.split(/(`[^`]+`)/).filter(Boolean).map((p) =>
    p.startsWith("`")
      ? `<tspan font-family="${MONO}" font-size="${(fs * 0.9).toFixed(1)}" fill="${C.tealDark}">${esc(p.slice(1, -1))}</tspan>`
      : `<tspan font-weight="${weight}" fill="${fill}">${esc(p)}</tspan>`).join("");
}

class Svg {
  constructor(w, h) { this.w = w; this.h = h; this.parts = []; }
  add(s) { this.parts.push(s); }
  rect(x, y, w, h, { fill = C.white, stroke = C.edge, sw = 1.2, r = 10, dash } = {}) {
    this.add(`<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${r}" fill="${fill}" stroke="${stroke}" stroke-width="${sw}"${dash ? ` stroke-dasharray="${dash}"` : ""}/>`);
  }
  text(x, y, s, { fs = 12, fill = C.ink, weight = 400, anchor = "start", mono = false } = {}) {
    this.add(`<text x="${x}" y="${y}" font-size="${fs}" font-weight="${weight}" fill="${fill}" text-anchor="${anchor}"${mono ? ` font-family="${MONO}"` : ""}>${runs(s, fs, fill, weight)}</text>`);
  }
  // wrapped paragraph, returns next y
  para(x, y, s, maxW, { fs = 11.5, fill = C.ink, lh = 1.45, bullet = false, weight = 400 } = {}) {
    const bw = bullet ? fs * 1.0 : 0;
    const lines = wrap(s, fs, maxW - bw);
    lines.forEach((ln, i) => {
      if (bullet && i === 0) this.add(`<circle cx="${x + 3}" cy="${y - fs * 0.34}" r="1.9" fill="${C.teal}"/>`);
      this.add(`<text x="${x + bw}" y="${y}" font-size="${fs}" fill="${fill}">${runs(ln, fs, fill, weight)}</text>`);
      y += fs * lh;
    });
    return y;
  }
  chip(x, y, label, { fill = C.pale, stroke = C.bandTeal, color = C.tealDark, fs = 10.5, mono = false, padX = 7, h = 19 } = {}) {
    const w = (mono ? [...label].length * fs * 0.62 : tw(label, fs)) + padX * 2;
    this.rect(x, y, w, h, { fill, stroke, r: h / 2, sw: 1 });
    this.add(`<text x="${x + padX}" y="${y + h / 2 + fs * 0.35}" font-size="${fs}" fill="${color}"${mono ? ` font-family="${MONO}"` : ""} font-weight="500">${esc(label)}</text>`);
    return x + w + 5;
  }
  arrow(x1, y1, x2, y2, { color = C.teal, sw = 1.8, dash } = {}) {
    const id = color.replace("#", "");
    this.add(`<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${color}" stroke-width="${sw}"${dash ? ` stroke-dasharray="${dash}"` : ""} marker-end="url(#ah${id})"/>`);
  }
  path(d, { color = C.orange, sw = 1.8, dash } = {}) {
    const id = color.replace("#", "");
    this.add(`<path d="${d}" fill="none" stroke="${color}" stroke-width="${sw}"${dash ? ` stroke-dasharray="${dash}"` : ""} marker-end="url(#ah${id})"/>`);
  }
  toString() {
    const markers = [C.teal, C.orange, C.ink, C.muted].map((c) => {
      const id = c.replace("#", "");
      return `<marker id="ah${id}" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="${c}"/></marker>`;
    }).join("");
    return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${this.w} ${this.h}" width="${this.w}" height="${this.h}" font-family="${SANS}"><defs>${markers}</defs><rect width="${this.w}" height="${this.h}" fill="#fff"/>${this.parts.join("")}</svg>\n`;
  }
}

// subscript helper: K_D -> text with tspans
const sub = (base, s) => `${base}<sub>${s}</sub>`;

// ---------- Card ----------
function card(g, { x, y, w, h, tag, title, role, accent = C.teal, band = C.bandTeal, inputs, rules, output, outChips }) {
  g.rect(x, y, w, h, { fill: C.white, stroke: C.edge, r: 12 });
  // header band
  g.add(`<path d="M${x},${y + 38} v-26 a12,12 0 0 1 12,-12 h${w - 24} a12,12 0 0 1 12,12 v26 z" fill="${band}"/>`);
  g.rect(x + 12, y + 9, 36, 20, { fill: accent, stroke: accent, r: 10, sw: 0 });
  g.text(x + 30, y + 23.5, tag, { fs: 11, fill: C.white, weight: 700, anchor: "middle" });
  g.text(x + 58, y + 24, title, { fs: 14, weight: 700, fill: C.ink });
  g.text(x + w - 12, y + 24, role, { fs: 10, fill: C.muted, anchor: "end" });

  let cy = y + 58;
  const px = x + 14, pw = w - 28;
  g.text(px, cy, "输入", { fs: 10.5, fill: accent, weight: 700 });
  let cx = px + 30;
  for (const ch of inputs) {
    const wch = tw(ch, 10.5) + 14;
    if (cx + wch > px + pw) { cx = px + 30; cy += 23; }
    cx = g.chip(cx, cy - 13.5, ch, { fill: C.gray, stroke: C.grayEdge, color: C.ink });
  }
  cy += 18;
  g.add(`<line x1="${px}" y1="${cy}" x2="${px + pw}" y2="${cy}" stroke="${C.edge}" stroke-width="1" stroke-dasharray="3 3"/>`);
  cy += 17;
  g.text(px, cy, "关键约束", { fs: 10.5, fill: accent, weight: 700 });
  cy += 17;
  for (const r of rules) cy = g.para(px, cy, r, pw, { fs: 11, bullet: true, lh: 1.42 }) + 2.5;

  // output box anchored at bottom, chips wrap
  const chipW = (ch) => (isMono(ch) ? [...ch].length * 10 * 0.62 : tw(ch, 10)) + 19;
  let rows = 1, rw0 = 0;
  for (const ch of outChips) { const cwd = chipW(ch); if (rw0 + cwd > pw - 20 && rw0 > 0) { rows++; rw0 = 0; } rw0 += cwd; }
  const oh = 30 + rows * 22;
  const oy = y + h - oh - 12;
  if (cy > oy - 4) console.warn(`overflow in card ${tag} ${title}: ${cy.toFixed(0)} > ${oy - 4}`);
  g.rect(px, oy, pw, oh, { fill: accent === C.orange ? C.warm : C.pale, stroke: band, r: 8 });
  g.text(px + 10, oy + 18, "输出", { fs: 10.5, fill: accent, weight: 700 });
  g.add(`<text x="${px + 42}" y="${oy + 18}" font-size="10.5" font-family="${MONO}" fill="${C.ink}" font-weight="500">${esc(output)}</text>`);
  let chx = px + 10, chy = oy + 26;
  for (const ch of outChips) {
    if (chx + chipW(ch) > px + pw - 6 && chx > px + 10) { chx = px + 10; chy += 22; }
    chx = g.chip(chx, chy, ch, { fill: C.white, stroke: band, color: accent === C.orange ? C.orange : C.tealDark, mono: isMono(ch), h: 17, fs: 10 });
  }
}

const isMono = (ch) => /^[A-Za-z_{}\[\]≤0-9.,() ]+$/.test(ch);

// ======================= Figure 1: prompt overview =======================
function figPrompts() {
  const g = new Svg(1000, 762);
  g.text(500, 26, "MedEvolve 核心提示词总览", { fs: 16, weight: 700, anchor: "middle" });
  g.text(500, 44, "每个角色的输入、关键约束与输出结构；提示词原文见下方各小节", { fs: 10.5, fill: C.muted, anchor: "middle" });

  const top = 62, H = 314, gapY = 24;
  const colW = 280, x1 = 20, x2 = 360, x3 = 700;

  card(g, { x: x1, y: top, w: colW, h: H, tag: "A.1", title: "数据探索", role: "数据探索 Agent",
    inputs: ["压缩摘要", "原始轨迹"],
    rules: ["压缩摘要不是原始证据；标记截断时只覆盖部分记录", "不能据摘要断言事实不存在",
      "归纳数据组织、时间变化、病程关联、证据引用与出题机会", "指导不含患者 ID、具体事实或来源 ID"],
    output: "DataExploration", outChips: ["guidance → K_D（患者无关）"] });
  card(g, { x: x1, y: top + H + gapY, w: colW, h: H, tag: "A.2", title: "环境探索", role: "环境探索 Agent",
    inputs: ["独立环境实例", "原生只读工具"],
    rules: ["交付前必须实际调用工具，不凭函数清单认定能力", "区分已验证能力、调用失败与未验证能力",
      "只查询，不写入、不模拟提交", "工具或环境故障不归纳为任务难度"],
    output: "EnvExploration", outChips: ["guidance → K_E（患者无关）"] });

  const midH = 2 * H + gapY;
  card(g, { x: x2, y: top, w: colW, h: midH, tag: "A.3", title: "任务生成", role: "任务生成角色", accent: C.orange, band: C.bandOrange,
    inputs: ["缓存轨迹", "K_D", "K_E", "策略 θ_r", "方向 Z（2 个）"],
    rules: ["仅使用传入轨迹的事实与来源 ID，无查询工具",
      "先从原始记录选择事实，再围绕事实构造题目、答案与证据",
      "`evidence` 每条 `statement` 只陈述该 `source_id` 实际支持的事实",
      "跨记录比较关系放入 `reference_answer`，`evidence` 分别陈述各条原始事实",
      "不把「轨迹未出现」推断为「未做／不存在／正常」",
      "证据不足时跳过该题位，允许输出空列表",
      "真实后续治疗不是决策题的唯一答案，接受有依据的等价方案",
      "两个方向位置各尝试一题，最多输出两题"],
    output: "TaskBatch", outChips: ["questions ≤ 2", "question", "reference_answer", "evidence[source_id, statement]"] });

  card(g, { x: x3, y: top, w: colW, h: H, tag: "A.4", title: "反馈分析", role: "反馈分析角色", accent: C.orange, band: C.bandOrange,
    inputs: ["核验结果", "解题结果", "难度评分"],
    rules: ["区分质量问题、临床难度、证据不足、题意歧义与运行故障",
      "工具／环境／模型故障、预算耗尽不等于难度提升",
      "无可靠评价时声明结论受限，不推断任务更难",
      "建议覆盖选题、证据组织、答案表述与生成方向"],
    output: "Feedback", outChips: ["反馈 F_r"] });
  card(g, { x: x3, y: top + H + gapY, w: colW, h: H, tag: "A.4", title: "策略更新", role: "策略更新角色", accent: C.orange, band: C.bandOrange,
    inputs: ["当前策略 θ_r", "反馈 F_r"],
    rules: ["只更新 `generation_guidance` 与 `allocation`；K_D、K_E 保持固定",
      "把有证据支持的经验沉淀为不绑定患者的自然语言指导",
      "个别题目的证据条数、作答格式不自动成为通用要求",
      "反馈证据不足时保留原策略，不靠加难度制造信号"],
    output: "Strategy", outChips: ["θ_{r+1} = (g_{r+1}, w_{r+1})"] });

  // arrows
  const ay1 = top + H / 2 - 20, ay2 = top + H + gapY + H / 2 + 20;
  g.arrow(x1 + colW + 2, ay1, x2 - 2, ay1);
  g.arrow(x1 + colW + 2, ay2, x2 - 2, ay2);
  g.text((x1 + colW + x2) / 2, ay1 - 6, "K_D", { fs: 10.5, fill: C.tealDark, weight: 700, anchor: "middle" });
  g.text((x1 + colW + x2) / 2, ay2 - 6, "K_E", { fs: 10.5, fill: C.tealDark, weight: 700, anchor: "middle" });

  // generation -> evaluation -> feedback
  g.arrow(x2 + colW + 2, top + 90, x3 - 2, top + 90, { color: C.ink });
  g.text((x2 + colW + x3) / 2, top + 72, "核验", { fs: 10.5, fill: C.ink, weight: 700, anchor: "middle" });
  g.text((x2 + colW + x3) / 2, top + 84, "解题", { fs: 10.5, fill: C.ink, weight: 700, anchor: "middle" });
  g.text((x2 + colW + x3) / 2, top + 112, "附录 B", { fs: 9.5, fill: C.muted, anchor: "middle" });
  // feedback -> update
  g.arrow(x3 + colW / 2, top + H + 1, x3 + colW / 2, top + H + gapY - 1, { color: C.orange });
  // update -> generation (loop)
  const ly = top + H + gapY + H / 2 + 20;
  const loopY = top + H + gapY + H / 2 + 20;
  g.path(`M${x3 - 2},${loopY} H${x2 + colW + 2}`, { color: C.orange, dash: "5 3" });
  g.text((x2 + colW + x3) / 2, loopY - 7, "θ_{r+1}", { fs: 10.5, fill: C.orange, weight: 700, anchor: "middle" });

  // legend
  const ly2 = 748;
  g.rect(20, ly2 - 12, 12, 12, { fill: C.bandTeal, stroke: C.teal, r: 3 }); g.text(38, ly2 - 2, "探索（固定，产出 K_D、K_E）", { fs: 10, fill: C.muted });
  g.rect(230, ly2 - 12, 12, 12, { fill: C.bandOrange, stroke: C.orange, r: 3 }); g.text(248, ly2 - 2, "生成与策略进化（逐轮更新）", { fs: 10, fill: C.muted });
  g.text(980, ly2 - 2, "θ_r = (g_r, w_r)；r 为进化轮次", { fs: 10, fill: C.muted, anchor: "end" });
  return g;
}

// ======================= Figure 2: verification & depth =======================
function figVerify() {
  const g = new Svg(1000, 610);
  g.text(500, 26, "证据核验与推理深度评价提示词", { fs: 16, weight: 700, anchor: "middle" });
  g.text(500, 44, "左：两阶段证据核验　右：推理深度 1–4 级与高阶题判定", { fs: 10.5, fill: C.muted, anchor: "middle" });

  // ---- left: verification ----
  const lx = 20, lw = 440, top = 62;
  g.rect(lx, top, lw, 530, { fill: C.white, stroke: C.edge, r: 12 });
  g.add(`<path d="M${lx},${top + 38} v-26 a12,12 0 0 1 12,-12 h${lw - 24} a12,12 0 0 1 12,12 v26 z" fill="${C.bandTeal}"/>`);
  g.text(lx + 16, top + 24, "证据核验", { fs: 14, weight: 700 });
  g.text(lx + lw - 14, top + 24, "先逐条，后整题", { fs: 10, fill: C.muted, anchor: "end" });

  const stage = (y, no, name, inputs, rules, outTxt) => {
    const px = lx + 16, pw = lw - 32;
    g.rect(px, y, 24, 24, { fill: C.teal, stroke: C.teal, r: 12, sw: 0 });
    g.text(px + 12, y + 16.5, String(no), { fs: 12, fill: C.white, weight: 700, anchor: "middle" });
    g.text(px + 34, y + 17, name, { fs: 13, weight: 700 });
    let cy = y + 40, cx = px;
    g.text(px, cy, "输入", { fs: 10.5, fill: C.teal, weight: 700 }); cx = px + 30;
    for (const ch of inputs) cx = g.chip(cx, cy - 13.5, ch, { fill: C.gray, stroke: C.grayEdge, color: C.ink, mono: /^[a-z_.]+$/.test(ch) });
    cy += 22;
    for (const r of rules) cy = g.para(px, cy, r, pw, { fs: 11.5, bullet: true, lh: 1.5 }) + 5;
    cy += 8;
    g.rect(px, cy - 6, pw, 26, { fill: C.pale, stroke: C.bandTeal, r: 7 });
    g.text(px + 10, cy + 11, "输出", { fs: 10.5, fill: C.teal, weight: 700 });
    g.add(`<text x="${px + 42}" y="${cy + 11}" font-size="10.5" font-family="${MONO}" fill="${C.ink}" font-weight="500">${esc(outTxt)}</text>`);
    return cy + 26;
  };
  let y = stage(top + 60, 1, "逐条证据声明核验", ["evidence.statement", "原始记录 resource"],
    ["只判断当前一条声明是否受当前这条原始记录支持",
      "逐项核对药物名称、剂量、检验值、时间等事实",
      "不得用其他记录或临床常识替代当前引用",
      "矛盾或没有记载时 valid=false"],
    "valid, reason（含 source_id 与具体事实）");
  g.arrow(lx + lw / 2, y + 6, lx + lw / 2, y + 40, { color: C.ink });
  g.text(lx + lw / 2 + 10, y + 28, "全部声明通过后", { fs: 9.5, fill: C.muted });
  stage(y + 54, 2, "整题核验", ["题目", "参考答案", "evidence_records"],
    ["核验题目与参考答案是否受原始数据支持，证据声明已逐条核验",
      "不推定记录未提供的患者事实",
      "决策点后的资料可支持答案；后续真实治疗不是唯一正确方案",
      "题目含糊、证据不足或答案不正确时 valid=false"],
    "valid, reason");

  // ---- right: depth ladder ----
  const rx = 480, rw = 500;
  g.rect(rx, top, rw, 530, { fill: C.white, stroke: C.edge, r: 12 });
  g.add(`<path d="M${rx},${top + 38} v-26 a12,12 0 0 1 12,-12 h${rw - 24} a12,12 0 0 1 12,12 v26 z" fill="${C.bandOrange}"/>`);
  g.text(rx + 16, top + 24, "推理深度评价", { fs: 14, weight: 700 });
  g.text(rx + rw - 14, top + 24, "按最简充分解题路径评分", { fs: 10, fill: C.muted, anchor: "end" });

  const levels = [
    [1, "事实检索", "读取、匹配、计数、汇总记录，或复述已明确记载的诊断。重复诊断不自动升级。", false],
    [2, "简单判断", "事实结合一条临床规则即可得出结论；并列多个独立简单判断不自动成为 3 分。", false],
    [3, "必要整合", "结论依赖多个独立事实间的关系：时间比较、竞争解释比较或条件推导；删去任一关键事实结论即变。", true],
    [4, "决策权衡", "在多个合理方案间做患者特异的选择：至少两个备选、冲突的收益风险，以及决定取舍的患者证据。", true],
  ];
  const sx = rx + 16, stepW = rw - 32, sh = 70;
  levels.forEach(([n, name, desc, ho], i) => {
    const sy = top + 52 + i * (sh + 8);
    const off = i * 18;
    g.rect(sx + off, sy, stepW - off, sh, { fill: ho ? C.warm : C.gray, stroke: ho ? C.bandOrange : C.grayEdge, r: 8 });
    g.rect(sx + off + 8, sy + 8, 26, 26, { fill: ho ? C.orange : C.muted, stroke: "none", r: 13, sw: 0 });
    g.text(sx + off + 21, sy + 26, String(n), { fs: 13, fill: C.white, weight: 700, anchor: "middle" });
    g.text(sx + off + 44, sy + 24, name, { fs: 12.5, weight: 700 });
    if (ho) g.chip(sx + stepW - 54, sy + 8, "高阶", { fill: C.orange, stroke: C.orange, color: C.white, h: 18, fs: 10 });
    g.para(sx + off + 44, sy + 41, desc, stepW - off - 56, { fs: 10.5, lh: 1.38 });
  });

  // shortcut check
  const cy0 = top + 52 + 4 * (sh + 8) + 2;
  g.rect(sx, cy0, stepW, 50, { fill: C.redPale, stroke: "#F2C7C3", r: 8 });
  g.text(sx + 10, cy0 + 16, "检查捷径", { fs: 11, weight: 700, fill: C.red });
  g.para(sx + 70, cy0 + 16, "题干是否已给出结论？诊断标签是否直接含答案？单条记录能否回答？是否只需通用建议？按仍需完成的实质推理评分；证据数量、跨度、答案长度不加分。", stepW - 82, { fs: 10, lh: 1.38 });

  // aggregation
  const ay = cy0 + 62;
  g.rect(sx, ay, stepW, 70, { fill: C.pale, stroke: C.bandTeal, r: 8 });
  g.chip(sx + 10, ay + 10, "GPT-4o-mini", { fill: C.white, stroke: C.bandTeal, color: C.ink, h: 20 });
  g.chip(sx + 10, ay + 36, "GPT-5.4-nano", { fill: C.white, stroke: C.bandTeal, color: C.ink, h: 20 });
  g.arrow(sx + 110, ay + 20, sx + 150, ay + 35);
  g.arrow(sx + 110, ay + 46, sx + 150, ay + 35);
  g.text(sx + 156, ay + 31, "取中位数", { fs: 11, weight: 700, fill: C.tealDark });
  g.text(sx + 156, ay + 46, "向下取整", { fs: 11, weight: 700, fill: C.tealDark });
  g.arrow(sx + 222, ay + 35, sx + 252, ay + 35);
  g.text(sx + 258, ay + 31, "聚合层级 l_i ≥ 3", { fs: 11, weight: 700, fill: C.ink });
  g.text(sx + 258, ay + 46, "且通过核验 → 高阶题", { fs: 11, weight: 700, fill: C.orange });
  g.text(sx + stepW - 10, ay + 58, "缺任一份有效评分则不形成聚合层级", { fs: 9.5, fill: C.muted, anchor: "end" });
  return g;
}


// ======================= Figure 3: A.5 case flow =======================
function figCase() {
  const g = new Svg(1000, 482);
  g.text(500, 26, "案例：一次证据核验失败如何转化为生成指导修订", { fs: 16, weight: 700, anchor: "middle" });
  g.text(500, 44, "PhysicianBench 进化第 1 轮，任务 0001:1:1", { fs: 10.5, fill: C.muted, anchor: "middle" });

  const top = 62, h = 232, w = 213, gap = 36, xs = [20, 269, 518, 767];
  const frame = (i, tag, title, accent, band) => {
    const x = xs[i];
    g.rect(x, top, w, h, { fill: C.white, stroke: C.edge, r: 12 });
    g.add(`<path d="M${x},${top + 38} v-26 a12,12 0 0 1 12,-12 h${w - 24} a12,12 0 0 1 12,12 v26 z" fill="${band}"/>`);
    g.rect(x + 12, top + 9, 22, 20, { fill: accent, stroke: accent, r: 10, sw: 0 });
    g.text(x + 23, top + 23.5, tag, { fs: 11, fill: C.white, weight: 700, anchor: "middle" });
    g.text(x + 44, top + 24, title, { fs: 14, weight: 700 });
    return x + 14;
  };
  const label = (x, y, t, c) => g.text(x, y, t, { fs: 10.5, fill: c, weight: 700 });

  // 1 task
  let x = frame(0, "1", "生成任务", C.teal, C.bandTeal), y = top + 60;
  label(x, y, "题目", C.teal); y += 16;
  y = g.para(x, y, "考虑患者在 2024 年 1 月 5 日和 1 月 21 日的实验室结果，评估其血糖和血脂水平的变化及对健康管理的影响。", w - 28, { fs: 11 }) + 8;
  label(x, y, "候选参考答案", C.teal); y += 16;
  g.rect(x - 4, y - 12, w - 20, 64, { fill: C.redPale, stroke: "#F2C7C3", r: 6 });
  g.para(x + 2, y + 2, "1 月 21 日的记录中未见进一步的血糖异常记录，说明在一定程度上患者的血糖得到控制。", w - 36, { fs: 11, fill: C.red });

  // 2 verify
  x = frame(1, "2", "证据核验", C.teal, C.bandTeal); y = top + 60;
  label(x, y, "任务声明", C.teal); y += 16;
  g.chip(x, y - 12, "Condition/2402", { fill: C.gray, stroke: C.grayEdge, color: C.ink, mono: true, fs: 10 });
  y += 24;
  y = g.para(x, y, "声明该条件为「高血糖」", w - 28, { fs: 11, fill: C.red }) + 8;
  label(x, y, "原始记录", C.teal); y += 16;
  y = g.para(x, y, "高脂血症（ICD-10-CM E78.5）", w - 28, { fs: 11 }) + 12;
  g.rect(x - 4, y - 4, w - 20, 40, { fill: C.redPale, stroke: "#F2C7C3", r: 8 });
  g.text(x + 6, y + 14, "✗  核验未通过", { fs: 12, fill: C.red, weight: 700 });
  g.add(`<text x="${x + 6}" y="${y + 30}" font-size="10" font-family="${MONO}" fill="${C.red}">valid=false</text>`);

  // 3 feedback
  x = frame(2, "3", "轮末反馈", C.orange, C.bandOrange); y = top + 60;
  label(x, y, "反馈结论", C.orange); y += 16;
  y = g.para(x, y, "缺失血糖复查结果，不能支持「血糖得到控制」。", w - 28, { fs: 11 }) + 10;
  label(x, y, "问题归类", C.orange); y += 14;
  let cx = g.chip(x, y, "证据不足", { fill: C.warm, stroke: C.bandOrange, color: C.orange });

  // 4 update
  x = frame(3, "4", "策略更新", C.orange, C.bandOrange); y = top + 60;
  label(x, y, "更新内容", C.orange); y += 16;
  y = g.para(x, y, "修订生成指导 g_r → g_{r+1}", w - 28, { fs: 11, bullet: true }) + 3;
  y = g.para(x, y, "检查与风险分层方向同步加限定", w - 28, { fs: 11, bullet: true }) + 3;
  y = g.para(x, y, "方向权重保持不变", w - 28, { fs: 11, bullet: true }) + 10;
  g.rect(x - 4, y - 4, w - 20, 40, { fill: C.warm, stroke: C.bandOrange, r: 8 });
  g.text(x + 6, y + 14, "θ_r → θ_{r+1}", { fs: 12, fill: C.orange, weight: 700 });
  g.text(x + 6, y + 30, "各方向权重仍为 1", { fs: 10, fill: C.muted });

  for (let i = 0; i < 3; i++) g.arrow(xs[i] + w + 3, top + 116, xs[i + 1] - 3, top + 116, { color: i === 0 ? C.ink : C.orange });
  g.text(xs[0] + w + gap / 2, top + 104, "核验", { fs: 9.5, fill: C.muted, anchor: "middle" });
  g.text(xs[1] + w + gap / 2, top + 104, "汇总", { fs: 9.5, fill: C.muted, anchor: "middle" });
  g.text(xs[2] + w + gap / 2, top + 104, "更新", { fs: 9.5, fill: C.muted, anchor: "middle" });

  // bottom: guidance panel
  const by = top + h + 36;
  g.arrow(xs[3] + w / 2, top + h + 2, xs[3] + w / 2, by - 3, { color: C.orange });
  g.rect(20, by, 960, 124, { fill: C.warm, stroke: C.bandOrange, r: 12 });
  g.text(36, by + 24, "更新后的生成指导 g_{r+1}（新增段落）", { fs: 13, weight: 700, fill: C.orange });
  g.rect(36, by + 38, 928, 34, { fill: C.white, stroke: C.bandOrange, r: 8 });
  g.add(`<rect x="36" y="${by + 38}" width="5" height="34" rx="2.5" fill="${C.orange}"/>`);
  g.para(54, by + 60, "纵向变化任务先确认可比数据：同一指标、同一单位、相近检测条件、明确基线与随访时间；缺失随访结果时不得推断「得到控制」「病情改善」或「恶化」。", 896, { fs: 12, lh: 1.5 });
  let bx = 36;
  const chips = [["检查与风险分层", "仅在基线与随访数据可比时整合，证据缺失时限定结论"], ["方向权重", "本轮各方向仍为 1"]];
  let by2 = by + 86;
  g.chip(bx, by2, chips[0][0], { fill: C.orange, stroke: C.orange, color: C.white, h: 20 });
  g.text(bx + tw(chips[0][0], 10.5) + 24, by2 + 14, chips[0][1], { fs: 11, fill: C.ink });
  g.chip(640, by2, chips[1][0], { fill: C.orange, stroke: C.orange, color: C.white, h: 20 });
  g.text(640 + tw(chips[1][0], 10.5) + 24, by2 + 14, chips[1][1], { fs: 11, fill: C.ink });
  g.text(980, 474, "红色标注：被核验或反馈指出的问题内容", { fs: 9.5, fill: C.muted, anchor: "end" });
  return g;
}

// subscript post-processing: K_D, K_E, θ_r, θ_{r+1}, g_{r+1}, w_{r+1}, F_r, l_i
function subscripts(svg) {
  const rep = [
    [/θ_\{r\+1\}/g, `θ<tspan font-size="70%" dy="3">r+1</tspan>`],
    [/g_\{r\+1\}/g, `g<tspan font-size="70%" dy="3">r+1</tspan>`],
    [/w_\{r\+1\}/g, `w<tspan font-size="70%" dy="3">r+1</tspan>`],
    [/([KθgwFl])_([DEri])(?![A-Za-z])/g, (m, a, b) => `${a}<tspan font-size="70%" dy="3">${b}</tspan>`],
  ];
  let out = svg;
  for (const [re, to] of rep) out = out.replace(re, to);
  // restore baseline after sub tspans (add dy back via empty tspan)
  out = out.replace(/(<tspan font-size="70%" dy="3">[^<]*<\/tspan>)/g, (m) => m.replace("</tspan>", "</tspan><tspan dy=\"-3\"></tspan>"));
  return out;
}

for (const [name, fig] of [["appendix-prompts-overview", figPrompts()], ["appendix-verification-depth", figVerify()], ["appendix-case-flow", figCase()]]) {
  writeFileSync(join(OUT, `${name}.svg`), subscripts(fig.toString()));
  console.log("wrote", name);
}
