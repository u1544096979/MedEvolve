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
    const markers = [C.teal, C.orange, C.ink, C.muted, ...(this.parts.some(p => p.includes("#000000")) ? ["#000000"] : [])].map((c) => {
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
function plainBox(g, x, y, w, h, title, paragraphs, fs = 16) {
  g.rect(x, y, w, h, { fill: "#FFFFFF", stroke: "#000000", sw: 1, r: 8 });
  g.text(x + 14, y + 27, title, { fs: 18, weight: 700, fill: "#000000" });
  let cy = y + 55;
  for (const text of paragraphs) {
    cy = g.para(x + 14, cy, text, w - 28, { fs, fill: "#000000", lh: 1.4 }) + 9;
  }
  if (cy - 9 > y + h - 8) throw new Error(`Text exceeds box: ${title} (${cy})`);
}

function figPrompts() {
  const g = new Svg(1000, 764);
  const top = 20, H = 350, gap = 24, w = 300;
  plainBox(g, 20, top, w, H, "数据探索", [
    "你是医疗任务数据探索 Agent。",
    "压缩摘要不是原始证据，标记为截断时只覆盖部分记录，不能据摘要遗漏判断事实不存在。",
    "归纳数据的组织方式、时间变化、病程关联、证据引用方式和可支持的出题机会。",
    "输出 DataExploration：guidance 是不绑定具体患者的抽象数据指导。不要复述患者 ID、具体事实或来源 ID；不要补充缓存未提供的事实。"
  ]);
  plainBox(g, 20, top + H + gap, w, H, "环境探索", [
    "你是医疗任务环境探索 Agent。",
    "通过原生工具了解查询名称、参数、返回结构及操作约束。仅访问开放的患者范围；只查询，不写入或模拟提交。",
    "交付前必须实际调用工具，不能仅凭函数清单认定能力已验证。区分已验证、调用失败与未验证能力。",
    "不要将具体患者观察或患者 ID 沉淀为公共指导；不要把工具或环境故障归纳为任务难度，或声称未验证能力已可用。"
  ]);
  plainBox(g, 350, top, w, 2 * H + gap, "任务生成", [
    "你是医疗任务生成角色。",
    "输入为患者缓存轨迹、固定的数据与环境探索指导、当前策略及两个生成方向。按两个位置的题型和难度各尝试生成一道任务。",
    "一次输出 TaskBatch，questions 最多两题；证据不足时允许少出题或空列表，不能补造。没有查询工具，仅使用传入轨迹的事实和来源 ID。",
    "先从 trajectory.sources 的原始记录选择事实，再构造题目、答案和证据。",
    "evidence 每条 statement 只陈述该 source_id 实际支持的事实；跨记录结论分别引用各条依据。比较关系放在 reference_answer 中，evidence 分别陈述原始事实。",
    "不能把轨迹未出现某项事实推断为该事实不存在、检查未做或结果正常；证据不足时跳过该题位。",
    "真实后续治疗不是临床决策题的唯一答案，允许有依据的合理等价方案。"
  ]);
  plainBox(g, 680, top, w, H, "反馈分析", [
    "你是医疗任务出题策略的反馈分析角色。",
    "区分质量问题、临床难度、证据不足、题意歧义及工具、环境、模型运行故障。",
    "工具、环境、模型故障及预算耗尽不能解释为难度提升；没有可靠完成评价时，明确结论受限，不能推断任务更难或解题失败。",
    "针对选题、证据组织、参考答案表述及生成方向，建议如何调整生成过程。"
  ]);
  plainBox(g, 680, top + H + gap, w, H, "策略更新", [
    "只根据当前 Strategy 与 Feedback 输出新的 Strategy。",
    "更新 generation_guidance 和 allocation，探索指导保持固定。经验须有证据支持、不绑定患者，并保留适用条件。",
    "个别题目的要求不自动成为通用规则。配比只安排两个生成位置，不改变数量。",
    "反馈证据不足时保留相应策略，不通过复杂题意、缺失数据或工具故障制造难度。"
  ]);
  return g;
}

// ======================= Figure 2: verification & depth =======================
function figVerify() {
  const g = new Svg(1000, 660);
  plainBox(g, 20, 20, 440, 280, "逐条证据声明核验", [
    "只判断当前一条证据声明是否受当前这条原始记录支持。resource 是按 evidence.source_id 读取的原始记录。",
    "逐项核对 statement 中的药物名称、剂量、检验值、时间等事实。只有记录支持全部声明时 valid=true；矛盾或没有记载时 valid=false。",
    "不得用其他记录或临床常识替代当前引用。reason 写明 source_id 和支持或冲突的具体事实。"
  ], 15.5);
  g.arrow(240, 303, 240, 333, { color: "#000000", sw: 1 });
  plainBox(g, 20, 336, 440, 304, "整题核验", [
    "核验题目和参考答案是否受 evidence_records 中的原始数据支持。记录已从环境读取，证据声明已逐条核验。",
    "不要推定记录未提供的患者事实。决策点后的资料可支持参考答案，但后续真实治疗并非临床决策题唯一正确方案。",
    "题目含糊、证据不足或参考答案不正确时返回 valid=false，并解释依据。"
  ], 15.5);
  plainBox(g, 480, 20, 500, 458, "推理深度评价", [
    "检查题干、证据和候选答案，按正确满足核心要求的最简充分解题路径评分；候选答案不自动正确，也不代表唯一解题路径。",
    "1分：事实检索。读取、匹配、计数、汇总或复述已记载的诊断；重复记录不自动升级。",
    "2分：简单判断。事实结合一条临床规则即可得出结论；多个独立简单判断不自动成为3分。",
    "3分：必要整合。结论依赖至少两个独立且不可替代的事实及其关系；删去关键事实后，结论改变或无法成立。",
    "4分：决策权衡。至少两个合理方案、相互冲突的收益风险或限制，以及决定取舍的患者证据。",
    "检查捷径：题干是否已给出结论、诊断标签是否含答案、单条记录能否回答、是否只需通用建议。证据多、跨度长、答案长本身不加分。",
    "reason 写明解题路径、必要事实及关系、捷径与层级依据；4分须说明方案与取舍。无法确认的病历事实不得补造。"
  ], 15.5);
  plainBox(g, 480, 502, 500, 138, "评分聚合与高阶题判定", [
    "GPT-4o-mini 与 GPT-5.4-nano 独立评分。两份有效评分取中位数并向下取整；缺任一份则不形成聚合层级。",
    "通过核验且聚合层级不低于3的任务计为高阶题。"
  ], 15.5);
  return g;
}


// ======================= Figure 3: A.5 case flow =======================
function figCase() {
  const g = new Svg(1000, 482);
  const top = 20, h = 282, w = 213, xs = [20, 269, 518, 767];
  plainBox(g, xs[0], top, w, h, "生成任务", [
    "题目：考虑患者在2024年1月5日和1月21日的实验室结果，评估其血糖和血脂水平的变化，以及对健康管理的潜在影响。请使用观察和诊断资料支持评估。",
    "候选答案：1月21日的记录中未见进一步的血糖异常记录，说明在一定程度上患者的血糖得到控制。"
  ], 13.5);
  plainBox(g, xs[1], top, w, h, "证据核验", [
    "任务声明：Condition/2402 为“高血糖”。",
    "原始记录：高脂血症（ICD-10-CM E78.5）。",
    "核验结果：未通过。原始记录不支持该证据声明。"
  ], 13.5);
  plainBox(g, xs[2], top, w, h, "轮末反馈", [
    "缺失血糖复查结果，不能支持“血糖得到控制”。"
  ], 13.5);
  plainBox(g, xs[3], top, w, h, "策略更新", [
    "检查与风险分层方向：仅在基线与随访数据可比时进行整合，证据缺失时限定结论。",
    "本轮各方向权重仍为1。"
  ], 13.5);
  for (let i = 0; i < 3; i++) g.arrow(xs[i] + w + 3, top + h / 2, xs[i + 1] - 3, top + h / 2, { color: "#000000", sw: 1 });
  const by = 338;
  g.arrow(xs[3] + w / 2, top + h + 2, xs[3] + w / 2, by - 3, { color: "#000000", sw: 1 });
  plainBox(g, 20, by, 960, 124, "更新后的生成指导", [
    "纵向变化任务先确认可比数据：同一指标、同一单位、相近检测条件、明确基线与随访时间；缺失随访结果时不得推断“得到控制”“病情改善”或“恶化”。"
  ], 16);
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
