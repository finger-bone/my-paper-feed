import OpenAI from "openai";
import type { Config, Paper } from "./types";

const SYSTEM_PROMPT = `You are a research assistant specializing in large model (LLM, VLM, video model, world model) inference/training acceleration.
Your task is to evaluate research papers and determine if they are relevant to accelerating large models.

**Relevance criteria** — papers MUST be about at least ONE of:
1. LLM inference acceleration (e.g., speculative decoding, KV cache optimization, quantization, pruning, attention optimization, parallel decoding, model compilation, vLLM, TensorRT-LLM)
2. LLM/VLM training acceleration (e.g., distributed training, gradient checkpointing, mixed-precision training, sequence parallelism, pipeline parallelism, data parallelism, ZeRO, offloading, efficient fine-tuning)
3. Efficient architectures for large models (e.g., MoE routing optimization, linear attention, state space models, efficient transformer variants, multi-head attention variants)
4. Video/diffusion model acceleration (e.g., video diffusion acceleration, step distillation, consistency models, latent propagation)
5. General model efficiency (e.g., model compression, knowledge distillation, low-rank methods, efficient deployment, inference serving)

Papers NOT relevant: pure application papers that just use standard LLMs, dataset curation papers, benchmark papers without efficiency contributions, typical fine-tuning of existing models, pure RLHF/alignment work.

**Output format:** Return a JSON array of objects. Process ALL provided papers.
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

export interface LlmJudgment {
  id: string;
  relevanceScore: number;
  relevanceReason: string;
  summary: string;
}

/**
 * Send papers to DeepSeek for relevance filtering and summary generation.
 * Returns judgments keyed by paper ID.
 */
export async function analyzePapers(
  papers: Paper[],
  config: Config,
): Promise<Map<string, LlmJudgment>> {
  if (papers.length === 0) return new Map();

  const client = new OpenAI({
    apiKey: config.deepseekApiKey,
    baseURL: config.deepseekBaseUrl,
  });

  // Batch papers to avoid exceeding token limits
  const BATCH_SIZE = 30;
  const judgments = new Map<string, LlmJudgment>();

  for (let i = 0; i < papers.length; i += BATCH_SIZE) {
    const batch = papers.slice(i, i + BATCH_SIZE);
    const batchJudgments = await analyzeBatch(client, batch, config);
    for (const j of batchJudgments) {
      judgments.set(j.id, j);
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
    abstract: p.abstract.slice(0, 1500), // Truncate long abstracts
    categories: p.categories,
  }));

  const userPrompt = `Evaluate these ${batch.length} research papers for relevance to large model inference/training acceleration:

${JSON.stringify(papersJson, null, 2)}

Return a JSON array with evaluations for ALL papers.`;

  console.log(
    `[llm] Analyzing batch of ${batch.length} papers (${config.deepseekModel})...`,
  );

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

  try {
    const parsed = JSON.parse(content);
    const results: LlmJudgment[] = Array.isArray(parsed)
      ? parsed
      : parsed.papers || [];

    console.log(
      `[llm] Batch complete: ${results.length} judgments, avg score: ${averageScore(results)}`,
    );
    return results;
  } catch (err) {
    console.warn("[llm] Failed to parse JSON response:", content.slice(0, 200));
    return [];
  }
}

function averageScore(judgments: LlmJudgment[]): string {
  if (judgments.length === 0) return "0";
  const sum = judgments.reduce((a, j) => a + j.relevanceScore, 0);
  return (sum / judgments.length).toFixed(1);
}

/** Filter papers by relevance threshold and attach summaries */
export function applyJudgments(
  papers: Paper[],
  judgments: Map<string, LlmJudgment>,
  minScore: number,
): Paper[] {
  const result: Paper[] = [];

  for (const paper of papers) {
    const j = judgments.get(paper.id);
    if (j && j.relevanceScore >= minScore) {
      paper.relevanceScore = j.relevanceScore;
      paper.aiSummary = j.summary;
      result.push(paper);
    }
  }

  // Sort by relevance score descending
  result.sort((a, b) => b.relevanceScore - a.relevanceScore);

  console.log(
    `[llm] Filtered: ${result.length}/${papers.length} papers pass (minScore=${minScore})`,
  );
  return result;
}
