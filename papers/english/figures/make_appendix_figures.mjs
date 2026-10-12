// Generate the English appendix SVGs with an embedded prompt font.
import { readFileSync, writeFileSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const out = dirname(fileURLToPath(import.meta.url));
const fontPath = execFileSync('kpsewhich', ['Inconsolatazi4-Regular.otf'], { encoding: 'utf8' }).trim();
const font = readFileSync(fontPath).toString('base64');
const escape = text => text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

function wrap(text, size, width) {
  const lines = [], words = text.split(/\s+/);
  let line = '';
  for (const word of words) {
    const next = line ? `${line} ${word}` : word;
    if (next.length * size * 0.5 > width && line) {
      lines.push(line);
      line = word;
    } else line = next;
    if (word.length * size * 0.5 > width) throw new Error(`Word exceeds box: ${word}`);
  }
  if (line) lines.push(line);
  return lines;
}

class Figure {
  constructor(width, height) { this.width = width; this.height = height; this.parts = []; }
  box(x, y, width, height, title, paragraphs, size = 17) {
    this.parts.push(`<rect x="${x}" y="${y}" width="${width}" height="${height}" rx="8" fill="#fff" stroke="#000" stroke-width="1"/>`);
    this.parts.push(`<text x="${x + 14}" y="${y + 28}" font-family="Arial,sans-serif" font-size="18" font-weight="700">${escape(title)}</text>`);
    let baseline = y + 54;
    for (const paragraph of paragraphs) {
      for (const line of wrap(paragraph, size, width - 28)) {
        this.parts.push(`<text x="${x + 14}" y="${baseline}" font-family="Inconsolata" font-size="${size}">${escape(line)}</text>`);
        baseline += size * 1.23;
      }
      baseline += 7;
    }
    if (baseline - size * 1.23 - 7 > y + height - 14) throw new Error(`Text exceeds box: ${title} (${baseline - y})`);
  }
  arrow(x1, y1, x2, y2) {
    this.parts.push(`<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="#000" stroke-width="1" marker-end="url(#arrow)"/>`);
  }
  svg() {
    return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${this.width} ${this.height}" width="${this.width}" height="${this.height}" fill="#000"><defs><style>@font-face{font-family:Inconsolata;src:url(data:font/otf;base64,${font}) format('opentype');font-weight:400}</style><marker id="arrow" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#000"/></marker></defs><rect width="${this.width}" height="${this.height}" fill="#fff"/>${this.parts.join('')}</svg>\n`;
  }
}

function prompts() {
  const f = new Figure(1000, 1024), h = 480, gap = 24;
  f.box(20, 20, 300, h, 'Data exploration', [
    'You are a medical task data exploration agent.',
    'Compressed summaries are not source evidence. Truncated summaries cover only part of the records; omissions do not establish absence.',
    'Identify data organization, temporal changes, clinical relationships, evidence citation patterns, and supported task opportunities.',
    'Return DataExploration. guidance must be abstract and patient-independent. Do not repeat patient IDs, specific facts, or source IDs, or add facts absent from the cache.'
  ]);
  f.box(20, 20 + h + gap, 300, h, 'Environment exploration', [
    'You are a medical task environment exploration agent.',
    'Use native tools to inspect queries, parameters, outputs, and constraints. Access only allowed patients. Query only; do not write or simulate submissions.',
    'Call tools before delivery; a function list does not verify capability. Distinguish verified, failed, and unverified capabilities.',
    'Do not put patient-specific observations or IDs in shared guidance. Do not treat tool or environment failures as task difficulty, or claim unverified capabilities are available.'
  ]);
  f.box(350, 20, 300, 2 * h + gap, 'Task generation', [
    'You are a medical task generator.',
    'Inputs: cached patient trajectory, fixed data and environment guidance, current strategy, and two generation directions. Attempt one task for each specified type and difficulty.',
    'Return one TaskBatch with at most two questions. Insufficient evidence permits fewer tasks or an empty list. Do not fabricate. No query tools are available; use only facts and source IDs in the trajectory.',
    'Select facts from original records in trajectory.sources before constructing the question, answer, and evidence.',
    'Each evidence statement must contain only facts supported by its source_id. Cite each source for cross-record conclusions. Put comparisons in reference_answer; state the source facts separately in evidence.',
    'An omitted fact does not mean it is absent, a test was not performed, or a result is normal. Skip a task slot when evidence is insufficient.',
    'Actual subsequent treatment is not the sole answer to a clinical decision task; accept evidence-supported equivalent alternatives.'
  ]);
  f.box(680, 20, 300, h, 'Feedback analysis', [
    'You analyze feedback on medical task generation strategies.',
    'Distinguish quality issues, clinical difficulty, insufficient evidence, ambiguity, and tool, environment, or model failures.',
    'Failures and exhausted budgets do not imply greater difficulty. Without reliable completion assessment, state the limits; do not infer greater difficulty or solver failure.',
    'Recommend changes to topic selection, evidence organization, reference answers, and generation directions.'
  ]);
  f.box(680, 20 + h + gap, 300, h, 'Strategy update', [
    'Return a new Strategy using only the current Strategy and Feedback.',
    'Update generation_guidance and allocation; keep exploration guidance fixed. Experience must be evidence-supported, patient-independent, and retain its applicability conditions.',
    'A task-specific requirement is not automatically a general rule. Allocation assigns two generation slots without changing their number.',
    'Keep the relevant strategy when feedback is insufficient. Do not create difficulty through convoluted wording, missing data, or tool failures.'
  ]);
  return f;
}

function verification() {
  const f = new Figure(1000, 930);
  f.box(20, 20, 440, 420, 'Evidence-claim verification', [
    'Judge only whether this evidence claim is supported by this source record. resource is the original record retrieved using evidence.source_id.',
    'Check each drug name, dose, test value, time, and other fact in statement. Set valid=true only if the record supports the entire claim; set valid=false if contradicted or undocumented.',
    'Do not substitute other records or clinical knowledge for this source. In reason, name the source_id and the supporting or conflicting facts.'
  ]);
  f.arrow(240, 443, 240, 461);
  f.box(20, 464, 440, 446, 'Whole-task verification', [
    'Verify that the question and reference answer are supported by the original data in evidence_records. Records have been retrieved from the environment, and evidence claims checked individually.',
    'Do not assume patient facts absent from the records. Post-decision data may support the reference answer, but actual subsequent treatment is not the only correct option for a clinical decision task.',
    'Return valid=false for an ambiguous question, insufficient evidence, or an incorrect reference answer, and explain the basis.'
  ]);
  f.box(480, 20, 500, 682, 'Reasoning-depth assessment', [
    'Inspect the question, evidence, and candidate answer. Score the simplest sufficient path that correctly meets the core requirements. A candidate answer is neither automatically correct nor the only solution path.',
    '1: Fact retrieval. Read, match, count, summarize, or restate recorded diagnoses. Repeated records do not raise the level.',
    '2: Simple judgment. Facts plus one clinical rule suffice. Multiple independent simple judgments do not imply level 3.',
    '3: Necessary integration. The conclusion depends on at least two independent, indispensable facts and their relationship. Removing a key fact changes or invalidates the conclusion.',
    '4: Decision trade-off. At least two reasonable options, conflicting benefits, risks, or constraints, and patient evidence that determines the choice.',
    'Check shortcuts: does the question state the conclusion, a diagnosis label contain the answer, one record suffice, or generic advice suffice? More evidence, longer time spans, or longer answers alone add no points.',
    'In reason, state the solution path, necessary facts and relationships, shortcuts, and level justification. For level 4, specify options and trade-offs. Do not fabricate unconfirmed patient facts.'
  ]);
  f.box(480, 726, 500, 184, 'Score aggregation and HO tasks', [
    'GPT-4o-mini and GPT-5.4-nano score independently. Take the median of two valid scores and round down. If either is missing, no aggregate level is formed.',
    'Verified tasks with aggregate level >=3 are higher-order (HO) tasks.'
  ]);
  return f;
}

function example() {
  const f = new Figure(1000, 666), top = 20, h = 442, w = 213, xs = [20, 269, 518, 767];
  f.box(xs[0], top, w, h, 'Generated task', [
    'Question: Using lab results from Jan 5 and Jan 21, 2024, assess changes in blood glucose and lipids and implications for health management. Support your assessment with observations and diagnoses.',
    'Candidate answer: No further glucose abnormalities were recorded on Jan 21, suggesting that glucose was controlled to some extent.'
  ], 16.5);
  f.box(xs[1], top, w, h, 'Evidence verification', [
    'Task claim: Condition/2402 is "hyperglycemia".',
    'Source record: hyperlipidemia (ICD-10-CM E78.5).',
    'Result: failed. The source record does not support this evidence claim.'
  ], 16.5);
  f.box(xs[2], top, w, h, 'Round feedback', [
    'Missing follow-up glucose results cannot support the claim that "glucose was controlled".'
  ], 16.5);
  f.box(xs[3], top, w, h, 'Strategy update', [
    'Testing and risk stratification: integrate baseline and follow-up data only when comparable; qualify conclusions when evidence is missing.',
    'All direction weights remain 1 in this round.'
  ], 16.5);
  for (let i = 0; i < 3; i++) f.arrow(xs[i] + w + 3, top + h / 2, xs[i + 1] - 3, top + h / 2);
  f.arrow(xs[3] + w / 2, top + h + 2, xs[3] + w / 2, 495);
  f.box(20, 498, 960, 148, 'Updated generation guidance', [
    'For longitudinal-change tasks, first confirm comparable data: the same measure and unit, similar testing conditions, and clear baseline and follow-up times. Without follow-up results, do not infer "controlled", "improved", or "worsened".'
  ]);
  return f;
}

for (const [name, figure] of [['appendix-prompts-overview', prompts()], ['appendix-case-flow', example()], ['appendix-verification-depth', verification()]]) {
  writeFileSync(join(out, `${name}.svg`), figure.svg());
}
