import OpenAI from "openai";
import type { Config, Paper } from "./types";

/*
 * ── Keyword pre-filtering ──
 *
 * Strategy: strict auto-include + broad auto-exclude.
 *   - auto-include: ONLY papers with obvious acceleration keywords
 *     (sparsity, quantization, distillation, acceleration, etc.)
 *   - auto-exclude: papers clearly about applications, platforms, or
 *     irrelevant domains (≥ 2 LOW_SIGNAL groups → penalty kicks in)
 *   - everything in between → LLM judges relevance + writes summary
 *
 *   score ≥ 7  → auto-include (obvious acceleration)
 *   score ≤ 3  → auto-exclude (irrelevant domain/application)
 *   4 ≤ score ≤ 6 → uncertain → LLM
 *
 * LOW_SIGNAL penalties are applied when ANY group matches (≥1),
 * capped at -3 total penalty.
 */

interface KeywordRule {
  /** Keywords that signal relevance (OR within same rule) */
  include: string[];
  /** Score boost if any include keyword matches */
  score: number;
}

/** "必然的" — obvious acceleration keywords (高置信度，直接放行) */
const HIGH_SIGNAL: KeywordRule[] = [
  // ── 推理加速 — 投机解码 / 并行生成 ──
  { include: ["speculative decoding", "speculative sampling", "draft model", "drafting"], score: 10 },
  { include: ["parallel decoding", "parallel generation", "jacobi decoding", "lookahead decoding"], score: 9 },
  { include: ["medusa", "eagle decoding", "stochastic speculative", "jump decoding"], score: 9 },
  { include: ["guided decoding", "constrained decoding", "predictive decoding"], score: 8 },
  { include: ["self-speculative", "speculative inference", "blockwise speculation", "multi-token prediction"], score: 9 },
  { include: ["multi-token decoding", "simultaneous decoding", "multi-token generation"], score: 9 },

  // ── 推理加速 — KV Cache ──
  { include: ["kv cache", "key-value cache", "kv compression", "kv quantization", "kv eviction"], score: 9 },
  { include: ["kv cache selection", "kv cache sharing", "cross-layer kv", "prefix caching"], score: 9 },
  { include: ["paged attention", "radix attention", "attention sink", "streaming llm"], score: 9 },
  { include: ["infini-attention", "ring attention", "blockwise parallel attention"], score: 8 },
  { include: ["prompt caching", "context caching", "semantic caching", "kv cache management"], score: 8 },
  { include: ["cache-aware", "cache-efficient", "cache reuse", "kv offloading"], score: 7 },

  // ── 推理加速 — 量化 ──
  { include: ["quantization", "quantized", "low-bit", "int4", "int8", "fp8", "fp4", "int2"], score: 8 },
  { include: ["weight quantization", "activation quantization", "smoothquant", "awq", "gptq"], score: 9 },
  { include: ["bitsandbytes", "nf4", "double quantization", "llm.int8"], score: 8 },
  { include: ["weight-only quantization", "dynamic quantization", "static quantization"], score: 8 },
  { include: ["w4a16", "w8a8", "w8a16", "w2a16", "normalfloat", "quip"], score: 9 },

  // ── 推理加速 — 编译 / Kernel ──
  { include: ["vllm", "tensorrt", "tensor-rt", "tensorrt-llm"], score: 9 },
  { include: ["model compilation", "kernel fusion", "triton", "cuda kernel"], score: 8 },
  { include: ["flash decoding", "flash-decoding", "flash attention 2", "flash attention 3"], score: 9 },
  { include: ["mlir", "xla", "iree", "tvm", "apache tvm", "tensor compiler"], score: 8 },
  { include: ["kernel optimization", "cuda graph", "kernel launch", "operator fusion"], score: 8 },

  // ── 推理加速 — 剪枝 / 稀疏 ──
  { include: ["pruning", "pruned", "sparsity", "sparse training", "sparse attention", "sparse model"], score: 8 },
  { include: ["structured pruning", "unstructured pruning", "magnitude pruning", "lottery ticket"], score: 8 },
  { include: ["activation sparsity", "dynamic sparsity", "mixture of sparsity", "weight sparsity"], score: 8 },
  { include: ["head pruning", "layer pruning", "depth pruning", "width pruning"], score: 7 },

  // ── 推理加速 — 连续批处理 / 服务 ──
  { include: ["continuous batching", "dynamic batching", "inference serving", "llm serving"], score: 7 },

  // ── 训练加速 — 分布式 / 并行策略 ──
  { include: ["sequence parallelism", "pipeline parallelism", "tensor parallelism", "data parallelism"], score: 8 },
  { include: ["expert parallelism", "context parallelism", "model parallelism"], score: 8 },
  { include: ["distributed training", "zero redundancy", "ZeRO", "fsdp", "fully sharded"], score: 8 },
  { include: ["deepseed", "megatron", "fairscale", "sharded data parallel"], score: 8 },
  { include: ["allreduce", "ring topology", "communication compression"], score: 7 },
  { include: ["communication efficient", "gradient compression", "gradient accumulation"], score: 7 },

  // ── 训练加速 — 显存优化 ──
  { include: ["gradient checkpointing", "activation checkpointing", "activation recomputation"], score: 8 },
  { include: ["mixed precision training", "bf16", "fp16 training", "amp training"], score: 7 },

  // ── 高效架构 — Attention ──
  { include: ["flash attention", "flash-attention", "efficient attention", "linear attention"], score: 9 },
  { include: ["multi-query attention", "mqa", "grouped query attention", "gqa"], score: 9 },
  { include: ["sliding window attention", "dilated attention", "local attention"], score: 8 },
  { include: ["linformer", "nystromformer", "performer", "reformer", "longformer", "bigbird"], score: 8 },
  { include: ["softmax attention", "attention approximation", "attention pruning", "attention sparsity"], score: 8 },
  { include: ["cross-attention optimization", "encoder-decoder acceleration"], score: 7 },

  // ── 高效架构 — MoE ──
  { include: ["mixture of experts", "mixture-of-experts", "moe", "expert routing", "expert balancing"], score: 9 },
  { include: ["soft moe", "hard moe", "switch transformer", "expert choice"], score: 8 },
  { include: ["routing optimization", "load balancing", "expert capacity"], score: 8 },
  { include: ["expert parallelism", "expert pruning", "expert merging", "fine-grained moe"], score: 9 },

  // ── 高效架构 — 状态空间模型 ──
  { include: ["state space model", "ssm", "mamba", "mamba 2"], score: 8 },
  { include: ["selective state space", "linear complexity", "linear rnn", "linear transformer"], score: 8 },
  { include: ["retentive network", "retnet", "recurrent attention", "hybrid architecture"], score: 8 },
  { include: ["linear attention", "efficient rnn", "gated linear rnn"], score: 8 },

  // ── 高效架构 — Transformer 变体 ──
  { include: ["efficient transformer", "transformer acceleration", "fast transformer"], score: 8 },

  // ── 模型压缩 / 蒸馏 ──
  { include: ["knowledge distillation", "distillation", "distilled", "self-distillation"], score: 7 },
  { include: ["model compression", "network compression", "model shrinking"], score: 8 },
  { include: ["logit distillation", "feature distillation", "structural distillation"], score: 7 },

  // ── 推理加速 — 早退 / 自适应 ──
  { include: ["early exiting", "early exit", "adaptive computation", "anycost"], score: 8 },
  { include: ["progressive decoding", "cascade", "cascaded inference"], score: 8 },
  { include: ["inference acceleration", "inference speedup", "inference optimization"], score: 9 },
  { include: ["training acceleration", "training speedup", "training optimization"], score: 8 },

  // ── 扩散 / 视频模型加速 ──
  { include: ["diffusion acceleration", "diffusion distillation", "step distillation"], score: 9 },
  { include: ["world model acceleration", "latent world model", "efficient world model"], score: 8 },
  { include: ["progressive distillation", "denoising acceleration", "ddim", "dpm-solver"], score: 8 },
  { include: ["diffusion transformer", "dit acceleration", "flow matching acceleration"], score: 9 },

  // ── 长上下文加速 ──
  { include: ["long context inference", "long context acceleration", "efficient long context"], score: 9 },

  // ── 硬件加速 ──
  { include: ["transformer accelerator", "neural accelerator", "tpu", "npu", "inference chip"], score: 8 },
];

/**
 * "应用/平台" — Keywords suggesting IRRELEVANCE (application, platform, etc.)
 * Applied when ANY group matches (≥1 group → -1 per group, cap -3).
 */
const LOW_SIGNAL: KeywordRule[] = [
  // ── 纯评测/数据/综述 ──
  { include: ["benchmark", "leaderboard", "evaluation suite"], score: -2 },
  { include: ["dataset", "corpus", "data collection", "data curation"], score: -2 },
  { include: ["survey", "review", "taxonomy", "overview"], score: -2 },

  // ── 纯应用领域（明显无关）──
  { include: ["clinical", "medical", "diagnosis", "healthcare"], score: -3 },
  { include: ["protein", "drug", "molecule", "biological"], score: -3 },
  { include: ["finance", "financial", "trading", "stock"], score: -3 },
  { include: ["law", "legal"], score: -3 },
  { include: ["agriculture", "crop", "weather forecast"], score: -3 },

  // ── 平台/框架/系统 ──
  { include: ["platform", "toolkit", "framework", "software system"], score: -2 },

  // ── RL / 控制 / 机器人 ──
  { include: ["reinforcement learning", "rlhf", "robot", "robotics"], score: -2 },
  { include: ["autonomous driving", "self-driving"], score: -2 },

  // ── 安全/对齐/解释性 ──
  { include: ["safety", "alignment", "bias", "fairness", "jailbreak"], score: -2 },
  { include: ["explainability", "interpretability", "xai"], score: -2 },

  // ── Agent / 工具 ──
  { include: ["agent", "tool use", "function calling"], score: -2 },
  { include: ["multi-agent", "agentic"], score: -2 },

  // ── NLP 下游应用 ──
  { include: ["multilingual", "machine translation"], score: -2 },
  { include: ["sentiment", "sentiment analysis"], score: -3 },
  { include: ["recommendation", "recommender system"], score: -3 },
  { include: ["code generation", "program synthesis"], score: -2 },
  { include: ["retrieval augmented generation", "rag"], score: -2 },
  { include: ["named entity recognition", "ner", "pos tagging"], score: -3 },
  { include: ["question answering", "qa", "reading comprehension"], score: -2 },

  // ── 对话/聊天应用 ──
  { include: ["chatbot", "conversational agent", "dialogue system", "chat system"], score: -2 },
  { include: ["customer service", "customer support", "helpdesk"], score: -2 },

  // ── 教育 ──
  { include: ["tutoring", "educational", "course", "learning system", "assessment", "curriculum", "educational assessment", "assessment design"], score: -2 },

  // ── 视觉推理 / VLM ──
  { include: ["visual reasoning", "visual question", "visual understanding", "visual recognition", "visual representation"], score: -2 },

  // ── 视频生成（非加速）──
  { include: ["video generation", "video synthesis", "video extrapolation", "video prediction", "video diffusion"], score: -3 },

  // ── 材料科学 ──
  { include: ["crystal", "materials science", "material generation", "chemistry", "molecular"], score: -3 },

  // ── RL 训练方法（非加速相关）──
  { include: ["grpo", "group relative policy optimization", "ppo", "proximal policy"], score: -2 },

  // ── 搜索/信息检索应用 ──
  { include: ["grep", "vector retrieval", "information access"], score: -2 },

  // ── 内容生成应用 ──
  { include: ["text-to-image", "text-to-video", "text-to-speech", "tts"], score: -2 },
  { include: ["music generation", "creative writing", "story generation"], score: -3 },
  { include: ["image generation", "image editing"], score: -2 },
  { include: ["content creation", "content generation", "social media"], score: -2 },

  // ── 其他应用 ──
  { include: ["information retrieval", "web search", "semantic search"], score: -2 },
  { include: ["email generation", "meeting summarization", "note-taking"], score: -2 },
  { include: ["e-commerce", "shopping", "product recommendation"], score: -2 },
  { include: ["video game", "gaming", "game playing"], score: -2 },
];

export interface LlmJudgment {
  id: string;
  relevanceScore: number;
  relevanceReason: string;
  summary: string;
}

export interface FilterThresholds {
  autoInclude: number;   // score ≥ this → auto-include (obvious acceleration)
  autoExclude: number;   // score ≤ this → auto-exclude (irrelevant)
}

const DEFAULT_THRESHOLDS: FilterThresholds = {
  autoInclude: 7,   // only obvious acceleration keywords auto-include
  autoExclude: 3,   // broader exclusion for application/platform papers
};

/**
 * Phase 1: Keyword-based pre-filtering.
 *
 *   score ≥ 7  → auto-include (obvious acceleration keywords matched)
 *   score ≤ 3  → auto-exclude (irrelevant domain/application)
 *   4 ≤ score ≤ 6 → uncertain → LLM
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

    // 1) High-signal: dominant acceleration keywords
    for (const rule of HIGH_SIGNAL) {
      for (const kw of rule.include) {
        if (text.includes(kw)) {
          score = Math.max(score, rule.score);
          break;
        }
      }
    }

    // 2) Low-signal: penalty when score hasn't reached auto-include
    //    ANY matching group → -1 per group (cap -3)
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
      if (lowMatchCount >= 1) {
        // -1 per group, cap at -3
        const penalty = Math.min(lowMatchCount, 3);
        score = Math.max(score - penalty, 0);
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

**CRITICAL: Be VERY strict. Default to EXCLUDING papers unless they clearly focus on acceleration.**

**Relevance criteria** — papers MUST be clearly about at least ONE of:
1. LLM inference acceleration (e.g., speculative decoding, KV cache optimization, quantization, pruning, attention optimization, parallel decoding, model compilation, vLLM, TensorRT-LLM)
2. LLM/VLM training acceleration (e.g., distributed training, gradient checkpointing, mixed-precision training, parallelism strategies, ZeRO, offloading, gradient compression)
3. Efficient architectures for large models (e.g., MoE routing optimization, linear attention, state space models, efficient transformer variants, grouped-query/multi-query attention)
4. Video/diffusion model acceleration (e.g., step distillation, flow matching acceleration, diffusion transformer acceleration)
5. General model efficiency (e.g., model compression, knowledge distillation, low-rank methods, inference serving, long-context optimization)

Papers ABSOLUTELY NOT relevant (score 0-2):
- Visual reasoning, visual question answering, or image/video generation papers that merely USE a model (no acceleration contribution)
- Papers about RL training methods (PPO, GRPO, RLHF) unless specifically applied to model efficiency
- Materials science, crystal generation, drug discovery, or other domain science
- Educational applications, assessment design, tutoring systems
- Agent/tool-use/search frameworks (e.g., "agentic search", "grep vs RAG")
- Benchmark/dataset papers without efficiency contributions
- Pure RLHF/alignment/safety work
- NLP applications like translation, sentiment analysis, QA without efficiency focus
- Platform/framework papers describing software systems without acceleration contributions
- Video generation, image generation papers unless they explicitly propose acceleration techniques
- Purely theoretical work with no demonstrated or claimed efficiency benefit
- Domain-specific fine-tuning of existing models for a vertical application

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
- summary: MUST be written in Chinese (中文), focus on the technical contribution
- Include ALL papers in the output array, even irrelevant ones (score 0-4 can have empty summary)
- Be very critical — err on the side of exclusion`;

/**
 * Phase 2: Send uncertain papers to DeepSeek for relevance analysis + summary.
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

/**
 * Phase 3: Generate Chinese summaries for keyword-decided papers.
 * These papers already passed keyword relevance check but have no summary.
 * Uses a lightweight batch prompt — much cheaper than relevance evaluation.
 */
export async function generateSummariesForDecided(
  papers: Paper[],
  config: Config,
  existingCache: Map<string, LlmJudgment>,
): Promise<Map<string, LlmJudgment>> {
  // Filter to papers with empty summaries
  const needsSummary = papers.filter((p) => {
    const cached = existingCache.get(p.id);
    return cached && cached.relevanceScore >= config.filterAutoInclude && !cached.summary;
  });

  if (needsSummary.length === 0) return new Map();

  console.log(
    `[llm] Generating Chinese summaries for ${needsSummary.length} keyword-decided papers...`,
  );

  const client = new OpenAI({
    apiKey: config.deepseekApiKey,
    baseURL: config.deepseekBaseUrl,
  });

  const BATCH_SIZE = 50;
  const summaryMap = new Map<string, LlmJudgment>();

  for (let i = 0; i < needsSummary.length; i += BATCH_SIZE) {
    const batch = needsSummary.slice(i, i + BATCH_SIZE);
    const summaries = await generateSummaryBatch(client, batch, config);
    for (const s of summaries) {
      summaryMap.set(s.id, s);
    }
  }

  return summaryMap;
}

async function generateSummaryBatch(
  client: OpenAI,
  batch: Paper[],
  config: Config,
): Promise<LlmJudgment[]> {
  const papersJson = batch.map((p) => ({
    id: p.id,
    title: p.title,
    abstract: p.abstract.slice(0, 1200),
  }));

  const prompt = `For each of the following research papers, write a 2-3 sentence Chinese summary (中文摘要) highlighting the key technical contribution. Focus on what technique they propose and what efficiency gain they achieve.

${JSON.stringify(papersJson, null, 2)}

Return a JSON object with a "papers" array:
{
  "papers": [
    {
      "id": "<paper-id>",
      "summary": "<2-3 sentence Chinese summary of the key contribution>"
    }
  ]
}

All summaries MUST be in Chinese (中文).`;

  try {
    const resp = await client.chat.completions.create({
      model: config.deepseekModel,
      messages: [
        { role: "system", content: "你是一个科研助手，用中文简要总结论文的核心贡献，每篇论文2-3句话。" },
        { role: "user", content: prompt },
      ],
      response_format: { type: "json_object" },
      temperature: 0.2,
      max_tokens: 4096,
    });

    const content = resp.choices?.[0]?.message?.content;
    if (!content) {
      console.warn("[llm-summary] Empty response, skipping batch");
      return [];
    }

    const parsed = JSON.parse(content);
    const results: Array<{ id: string; summary: string }> = Array.isArray(parsed)
      ? parsed
      : parsed.papers || [];

    return results.map((r) => ({
      id: r.id,
      relevanceScore: 0, // not used — these are already deemed relevant
      relevanceReason: "",
      summary: r.summary || "",
    }));
  } catch (err) {
    console.warn("[llm-summary] LLM call failed for summary batch:", err);
    return [];
  }
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
