import OpenAI from "openai";
import type { Config, Paper } from "./types";

/*
 * ── Keyword pre-filtering ──
 *
 * Strategy: be lenient — prefer sending borderline papers to the LLM
 * over filtering them out prematurely.
 *
 *   score ≥ 7  → auto-include (skip LLM)
 *   score ≤ 2  → auto-exclude (skip LLM)
 *   3 ≤ score ≤ 6 → uncertain → send to LLM
 *
 * Low-signal penalties are weak and only kick in when ≥ 2 different
 * low-signal rule groups match. This prevents a single stray keyword
 * from tanking an otherwise relevant paper.
 */

interface KeywordRule {
  /** Keywords that signal relevance (OR within same rule) */
  include: string[];
  /** Score boost if any include keyword matches */
  score: number;
}

/** High-confidence keywords → auto-include */
const HIGH_SIGNAL: KeywordRule[] = [
  // ── 推理加速 — 投机解码 / 并行生成 ──
  { include: ["speculative decoding", "speculative sampling", "draft model", "drafting"], score: 10 },
  { include: ["parallel decoding", "parallel generation", "jacobi decoding", "lookahead decoding"], score: 9 },
  { include: ["medusa", "eagle decoding", "stochastic speculative", "jump decoding"], score: 9 },
  { include: ["guided decoding", "constrained decoding", "predictive decoding"], score: 8 },

  // ── 推理加速 — KV Cache ──
  { include: ["kv cache", "key-value cache", "kv compression", "kv quantization", "kv eviction"], score: 9 },
  { include: ["kv cache selection", "kv cache sharing", "cross-layer kv", "prefix caching"], score: 9 },
  { include: ["paged attention", "radix attention", "attention sink", "streaming llm"], score: 9 },
  { include: ["infini-attention", "ring attention", "blockwise parallel attention"], score: 8 },

  // ── 推理加速 — 量化 ──
  { include: ["quantization", "quantized", "low-bit", "int4", "int8", "fp8", "fp4", "int2"], score: 8 },
  { include: ["weight quantization", "activation quantization", "smoothquant", "awq", "gptq"], score: 9 },
  { include: ["bitsandbytes", "nf4", "double quantization", "llm.int8"], score: 8 },

  // ── 推理加速 — 编译 / Kernel ──
  { include: ["vllm", "tensorrt", "tensor-rt", "tensorrt-llm"], score: 9 },
  { include: ["model compilation", "kernel fusion", "triton", "cuda kernel"], score: 8 },
  { include: ["flash decoding", "flash-decoding", "flash attention 2", "flash attention 3"], score: 9 },

  // ── 推理加速 — 剪枝 / 稀疏 ──
  { include: ["pruning", "pruned", "sparsity", "sparse training", "sparse attention", "sparse model"], score: 8 },
  { include: ["structured pruning", "unstructured pruning", "magnitude pruning", "lottery ticket"], score: 8 },

  // ── 推理加速 — 连续批处理 / 服务 ──
  { include: ["continuous batching", "dynamic batching", "inference serving", "llm serving"], score: 8 },
  { include: ["llama.cpp", "ggml", "gguf", "ollama", "tgi", "text generation inference"], score: 7 },
  { include: ["onnx", "onnx runtime", "openvino", "cpu inference", "edge deployment"], score: 7 },

  // ── 训练加速 — 分布式 / 并行策略 ──
  { include: ["sequence parallelism", "pipeline parallelism", "tensor parallelism", "data parallelism"], score: 8 },
  { include: ["expert parallelism", "context parallelism", "model parallelism"], score: 8 },
  { include: ["distributed training", "zero redundancy", "ZeRO", "fsdp", "fully sharded"], score: 8 },
  { include: ["deepseed", "megatron", "fairscale", "sharded data parallel"], score: 8 },
  { include: ["allreduce", "ring topology", "communication compression"], score: 7 },

  // ── 训练加速 — 显存优化 ──
  { include: ["gradient checkpointing", "activation checkpointing", "activation recomputation"], score: 8 },
  { include: ["offloading", "memory efficient", "memory optimization", "vram optimization"], score: 7 },
  { include: ["mixed precision training", "bf16", "fp16 training", "amp training"], score: 7 },

  // ── 训练加速 — 数据 / 调度 ──
  { include: ["dataset pruning", "data selection", "data filtering", "curriculum learning"], score: 7 },
  { include: ["progressive training", "stagewise training", "warmup"], score: 6 },
  { include: ["gradient accumulation", "gradient compression", "gradient merging"], score: 7 },

  // ── 高效架构 — Attention ──
  { include: ["flash attention", "flash-attention", "efficient attention", "linear attention"], score: 9 },
  { include: ["multi-query attention", "mqa", "grouped query attention", "gqa"], score: 9 },
  { include: ["sliding window attention", "dilated attention", "local attention"], score: 8 },
  { include: ["linformer", "nystromformer", "performer", "reformer", "longformer", "bigbird"], score: 8 },
  { include: ["transformer-xl", "compressive transformer", "adaptive attention"], score: 7 },

  // ── 高效架构 — MoE ──
  { include: ["mixture of experts", "mixture-of-experts", "moe", "expert routing", "expert balancing"], score: 9 },
  { include: ["soft moe", "hard moe", "switch transformer", "expert choice"], score: 8 },
  { include: ["routing optimization", "load balancing", "expert capacity"], score: 8 },

  // ── 高效架构 — 状态空间模型 ──
  { include: ["state space model", "ssm", "mamba", "mamba 2"], score: 8 },
  { include: ["selective state space", "linear complexity", "linear rnn", "linear transformer"], score: 8 },
  { include: ["retentive network", "retnet", "recurrent attention", "hybrid architecture"], score: 8 },

  // ── 高效架构 — Transformer 变体 ──
  { include: ["efficient transformer", "transformer acceleration", "fast transformer"], score: 8 },
  { include: ["funnel transformer", "fnet", "sinkhorn transformer", "synthesizer"], score: 7 },

  // ── 模型压缩 / 蒸馏 ──
  { include: ["knowledge distillation", "distillation", "distilled", "self-distillation"], score: 7 },
  { include: ["model compression", "network compression", "model shrinking"], score: 8 },
  { include: ["low-rank", "lora", "low-rank adaptation", "low rank approximation"], score: 7 },
  { include: ["parameter-efficient fine-tuning", "peft", "adapter", "prefix tuning", "prompt tuning"], score: 7 },

  // ── 推理加速 — 早退 / 自适应 ──
  { include: ["early exiting", "early exit", "adaptive computation", "anycost"], score: 8 },
  { include: ["progressive decoding", "cascade", "cascaded inference"], score: 8 },
  { include: ["inference acceleration", "inference speedup", "inference optimization"], score: 9 },
  { include: ["training acceleration", "training speedup", "training optimization"], score: 8 },

  // ── 扩散 / 视频模型加速 ──
  { include: ["diffusion acceleration", "diffusion distillation", "step distillation"], score: 9 },
  { include: ["consistency model", "consistency distillation", "latent propagation"], score: 9 },
  { include: ["efficient video", "video transformer acceleration", "video diffusion acceleration"], score: 8 },
  { include: ["world model acceleration", "latent world model", "efficient world model"], score: 8 },
  { include: ["progressive distillation", "denoising acceleration", "ddim", "dpm-solver"], score: 8 },

  // ── 通用效率 ──
  { include: ["throughput", "latency", "model efficiency", "compute efficient"], score: 6 },
  { include: ["efficient", "efficiency"], score: 5 },
  { include: ["inference scaling", "test-time compute", "inference compute"], score: 7 },
];

/** Keywords that suggest IRRELEVANCE. Penalties are mild so a single
 *  stray keyword won't sink a paper. Only applied when ≥ 2 different
 *  rule groups match. */
const LOW_SIGNAL: KeywordRule[] = [
  // ── 评测/数据/综述 ──
  { include: ["benchmark", "leaderboard", "evaluation suite"], score: -2 },
  { include: ["dataset", "corpus", "data collection", "data curation"], score: -2 },
  { include: ["survey", "review", "taxonomy", "overview"], score: -2 },

  // ── 无关应用领域 ──
  { include: ["clinical", "medical", "diagnosis", "healthcare"], score: -3 },
  { include: ["protein", "drug", "molecule", "biological"], score: -3 },
  { include: ["finance", "financial", "trading", "stock"], score: -3 },
  { include: ["law", "legal"], score: -3 },

  // ── RL / 控制 / 机器人 ──
  { include: ["reinforcement learning", "rlhf", "robot", "robotics"], score: -2 },
  { include: ["autonomous driving", "self-driving"], score: -2 },

  // ── 安全/对齐/解释性 ──
  { include: ["safety", "alignment", "bias", "fairness", "jailbreak"], score: -2 },
  { include: ["explainability", "interpretability", "xai"], score: -2 },

  // ── Agent / 工具 ──
  { include: ["agent", "tool use", "function calling"], score: -2 },

  // ── NLP 应用 ──
  { include: ["multilingual", "machine translation"], score: -2 },
  { include: ["sentiment", "sentiment analysis"], score: -3 },
  { include: ["recommendation", "recommender system"], score: -3 },
  { include: ["code generation", "program synthesis"], score: -2 },
  { include: ["retrieval augmented generation", "rag"], score: -2 },

  // ── 纯生成应用 ──
  { include: ["text-to-image", "text-to-video", "text-to-speech", "tts"], score: -2 },
  { include: ["music generation", "creative writing"], score: -3 },
];

export interface LlmJudgment {
  id: string;
  relevanceScore: number;
  relevanceReason: string;
  summary: string;
}

export interface FilterThresholds {
  autoInclude: number;   // score ≥ this → skip LLM, mark relevant
  autoExclude: number;   // score ≤ this → skip LLM, mark irrelevant
}

const DEFAULT_THRESHOLDS: FilterThresholds = {
  autoInclude: 7,   // was 8 — more papers auto-included
  autoExclude: 2,   // was 3 — fewer papers auto-excluded
};

/**
 * Phase 1: Keyword-based pre-filtering.
 * Lenient strategy — prefers sending to LLM over premature exclusion.
 */
export function keywordFilter(
  papers: Paper[],
  thresholds: FilterThresholds = DEFAULT_THRESHOLDS,
): { decided: Map<string, LlmJudgment>; uncertain: Paper[] } {
  const decided = new Map<string, LlmJudgment>();
  const uncertain: Paper[] = [];

  for (const paper of papers) {
    const text = `${paper.title}\n${paper.abstract}`.toLowerCase();
    let score = 5; // neutral baseline

    // 1) High-signal keywords boost score
    for (const rule of HIGH_SIGNAL) {
      for (const kw of rule.include) {
        if (text.includes(kw)) {
          score = Math.max(score, rule.score);
          break;
        }
      }
    }

    // 2) Low-signal penalty: only when score is not already high
    //    AND at least 2 different rule groups match (avoids single-kw tanking)
    if (score < thresholds.autoInclude) {
      let lowMatchCount = 0;
      for (const rule of LOW_SIGNAL) {
        for (const kw of rule.include) {
          if (text.includes(kw)) {
            lowMatchCount++;
            break;
          }
        }
      }
      if (lowMatchCount >= 2) {
        // Mild penalty capped at -3 total regardless of how many matched
        const penalty = Math.min(lowMatchCount * -1, -3);
        score = Math.max(score + penalty, 0);
      }
    }

    score = Math.max(0, Math.min(10, score));

    // 3) Classify
    if (score >= thresholds.autoInclude) {
      decided.set(paper.id, {
        id: paper.id,
        relevanceScore: score,
        relevanceReason: "Keyword match: relevant terms detected",
        summary: "",
      });
    } else if (score <= thresholds.autoExclude) {
      decided.set(paper.id, {
        id: paper.id,
        relevanceScore: score,
        relevanceReason: "Keyword match: low-relevance terms detected",
        summary: "",
      });
    } else {
      uncertain.push(paper);
    }
  }

  const autoIncluded = [...decided.values()].filter((j) => j.relevanceScore >= thresholds.autoInclude).length;
  const autoExcluded = [...decided.values()].filter((j) => j.relevanceScore <= thresholds.autoExclude).length;

  console.log(
    `[filter] Keyword pre-filter: ${decided.size} decided ` +
    `(${autoIncluded} auto-include, ${autoExcluded} auto-exclude), ` +
    `${uncertain.length} uncertain → LLM`,
  );
  return { decided, uncertain };
}

/*
 * ── LLM analysis ──
 */

const SYSTEM_PROMPT = `You are a research assistant specializing in large model (LLM, VLM, video model, world model) inference/training acceleration.
Your task is to evaluate research papers and determine if they are relevant to accelerating large models.

**Relevance criteria** — papers MUST be about at least ONE of:
1. LLM inference acceleration (e.g., speculative decoding, KV cache optimization, quantization, pruning, attention optimization, parallel decoding, model compilation, vLLM, TensorRT-LLM)
2. LLM/VLM training acceleration (e.g., distributed training, gradient checkpointing, mixed-precision training, sequence parallelism, pipeline parallelism, data parallelism, ZeRO, offloading, efficient fine-tuning)
3. Efficient architectures for large models (e.g., MoE routing optimization, linear attention, state space models, efficient transformer variants, multi-head attention variants)
4. Video/diffusion model acceleration (e.g., video diffusion acceleration, step distillation, consistency models, latent propagation)
5. General model efficiency (e.g., model compression, knowledge distillation, low-rank methods, efficient deployment, inference serving)

Papers NOT relevant: pure application papers that just use standard LLMs, dataset curation papers, benchmark papers without efficiency contributions, typical fine-tuning of existing models, pure RLHF/alignment work.

**Output format:** Return a JSON object with a "papers" array:
{
  "papers": [
    {
      "id": "<paper-id>",
      "relevanceScore": <0-10 integer>,
      "relevanceReason": "<1-sentence explanation>",
      "summary": "<2-3 sentence Chinese summary highlighting the key contribution>"
    }
  ]
}

- relevanceScore: 0 = completely irrelevant, 10 = perfectly relevant
- summary: MUST be written in Chinese (中文)
- Include ALL papers in the output array, even irrelevant ones (score 0-4 can have empty summary)
- Be critical — not every paper mentioning "efficient" is actually about model acceleration`;

/**
 * Phase 2: Send uncertain papers to DeepSeek for deeper analysis.
 * Checks cache first and skips already-judged papers.
 */
export async function analyzeUncertainPapers(
  uncertain: Paper[],
  config: Config,
  cache: Map<string, LlmJudgment>,
): Promise<Map<string, LlmJudgment>> {
  if (uncertain.length === 0) return new Map();

  // Filter out papers already in cache
  const needsLlm = uncertain.filter((p) => !cache.has(p.id));
  const fromCache = uncertain.filter((p) => cache.has(p.id));
  console.log(
    `[llm] ${uncertain.length} uncertain papers: ${fromCache.length} from cache, ${needsLlm.length} need LLM`,
  );

  if (needsLlm.length === 0) {
    const result = new Map<string, LlmJudgment>();
    for (const p of uncertain) {
      const cached = cache.get(p.id);
      if (cached) result.set(p.id, cached);
    }
    return result;
  }

  const client = new OpenAI({
    apiKey: config.deepseekApiKey,
    baseURL: config.deepseekBaseUrl,
  });

  const BATCH_SIZE = 30;
  const judgments = new Map<string, LlmJudgment>();

  for (let i = 0; i < needsLlm.length; i += BATCH_SIZE) {
    const batch = needsLlm.slice(i, i + BATCH_SIZE);
    const batchJudgments = await analyzeBatch(client, batch, config);
    for (const j of batchJudgments) {
      if (j) judgments.set(j.id, j);
    }
  }

  return judgments;
}

async function analyzeBatch(
  client: OpenAI,
  batch: Paper[],
  config: Config,
): Promise<LlmJudgment[]> {
  const papersJson = batch.map((p) => ({
    id: p.id,
    title: p.title,
    abstract: p.abstract.slice(0, 1500),
    categories: p.categories,
  }));

  const userPrompt = `Evaluate these ${batch.length} research papers for relevance to large model inference/training acceleration:

${JSON.stringify(papersJson, null, 2)}

Return a JSON object with a "papers" array containing evaluations for ALL papers.`;

  console.log(
    `[llm] Analyzing batch of ${batch.length} papers (${config.deepseekModel})...`,
  );

  try {
    const resp = await client.chat.completions.create({
      model: config.deepseekModel,
      messages: [
        { role: "system", content: SYSTEM_PROMPT },
        { role: "user", content: userPrompt },
      ],
      response_format: { type: "json_object" },
      temperature: 0.1,
      max_tokens: 4096,
    });

    const content = resp.choices?.[0]?.message?.content;
    if (!content) {
      console.warn("[llm] Empty response, skipping batch");
      return [];
    }

    const parsed = JSON.parse(content);
    const results: LlmJudgment[] = Array.isArray(parsed)
      ? parsed
      : parsed.papers || [];

    console.log(
      `[llm] Batch complete: ${results.length} judgments, avg score: ${averageScore(results)}`,
    );
    return results;
  } catch (err) {
    console.warn("[llm] LLM call failed for batch, will retry next run:", err);
    return [];
  }
}

function averageScore(judgments: LlmJudgment[]): string {
  if (judgments.length === 0) return "0";
  const sum = judgments.reduce((a, j) => a + j.relevanceScore, 0);
  return (sum / judgments.length).toFixed(1);
}

/** Merge decided + LLM judgments, filter by threshold, sort */
export function applyJudgments(
  papers: Paper[],
  allJudgments: Map<string, LlmJudgment>,
  minScore: number,
): Paper[] {
  const result: Paper[] = [];

  for (const paper of papers) {
    const j = allJudgments.get(paper.id);
    if (j && j.relevanceScore >= minScore) {
      paper.relevanceScore = j.relevanceScore;
      paper.aiSummary = j.summary || "";
      result.push(paper);
    }
  }

  result.sort((a, b) => b.relevanceScore - a.relevanceScore);

  console.log(
    `[filter] Final: ${result.length}/${papers.length} papers pass (minScore=${minScore})`,
  );
  return result;
}
