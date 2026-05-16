import { loadConfig, type Config, type Paper } from "./types";
import { fetchArxivPapers } from "./sources/arxiv";
import { fetchHuggingFacePapers } from "./sources/huggingface";
import { analyzePapers, applyJudgments } from "./llm";
import { generateRssFeed } from "./rss";

async function main() {
  console.log("=".repeat(60));
  console.log("Paper Feed — LLM Acceleration Papers Curator");
  console.log("=".repeat(60));

  const config = loadConfig();
  validateConfig(config);

  // Phase 1: Fetch papers from all sources
  console.log("\n📥 Phase 1: Fetching papers...");
  const allPapers = await fetchAllSources(config);
  console.log(`Total papers fetched: ${allPapers.length}`);

  if (allPapers.length === 0) {
    console.log("No papers found. Exiting.");
    return;
  }

  // Phase 2: Analyze with DeepSeek
  console.log("\n🤖 Phase 2: Analyzing with DeepSeek...");
  const judgments = await analyzePapers(allPapers, config);

  // Phase 3: Filter and sort
  console.log("\n📋 Phase 3: Filtering by relevance...");
  const relevantPapers = applyJudgments(allPapers, judgments, config.minRelevanceScore);

  if (relevantPapers.length === 0) {
    console.log("No relevant papers found. Generating empty feed.");
  } else {
    console.log(`\n📄 Relevant papers (score >= ${config.minRelevanceScore}):`);
    for (const p of relevantPapers) {
      console.log(`  [${p.relevanceScore}/10] ${p.title}`);
      console.log(`         ${p.aiSummary.slice(0, 120)}...`);
      console.log(`         ${p.link}`);
      console.log();
    }
  }

  // Phase 4: Generate RSS
  console.log("\n📰 Phase 4: Generating RSS feed...");
  generateRssFeed(relevantPapers, config);

  // Phase 5: Print summary
  console.log("\n" + "=".repeat(60));
  console.log("✅ Done!");
  console.log(`   Total papers fetched: ${allPapers.length}`);
  console.log(`   Relevant papers: ${relevantPapers.length}`);
  if (relevantPapers.length > 0) {
    console.log(
      `   Average relevance: ${(
        relevantPapers.reduce((s, p) => s + p.relevanceScore, 0) /
        relevantPapers.length
      ).toFixed(1)}/10`,
    );
  }
  console.log(`   Output: ${config.outputPath}`);
  console.log("=".repeat(60));
}

async function fetchAllSources(config: Config): Promise<Paper[]> {
  const sources = [
    { fetcher: fetchArxivPapers, name: "arXiv" },
    { fetcher: fetchHuggingFacePapers, name: "HuggingFace" },
  ] as const;

  const results = await Promise.allSettled(
    sources.map((s) => s.fetcher(config.arxivMaxResults, config.lookbackDays)),
  );

  const papers: Paper[] = [];
  const seen = new Set<string>();

  for (let i = 0; i < results.length; i++) {
    const result = results[i]!;
    const sourceName = sources[i]!.name;

    if (result.status === "fulfilled") {
      for (const paper of result.value) {
        if (!seen.has(paper.id)) {
          seen.add(paper.id);
          papers.push(paper);
        }
      }
      console.log(`[fetch] ${sourceName}: ${result.value.length} papers`);
    } else {
      console.error(`[fetch] ${sourceName} failed:`, result.reason);
    }
  }

  return papers;
}

function validateConfig(config: Config): void {
  if (!config.deepseekApiKey) {
    console.error(
      "❌ DEEPSEEK_API_KEY environment variable is required.\n" +
        "   Set it in GitHub Secrets or your local .env file.",
    );
    process.exit(1);
  }

  if (config.lookbackDays < 1 || config.lookbackDays > 30) {
    console.warn("⚠️  LOOKBACK_DAYS should be between 1 and 30. Using 7.");
    config.lookbackDays = 7;
  }

  if (config.minRelevanceScore < 0 || config.minRelevanceScore > 10) {
    console.warn("⚠️  MIN_RELEVANCE_SCORE should be 0-10. Using 5.");
    config.minRelevanceScore = 5;
  }
}

main().catch((err) => {
  console.error("❌ Fatal error:", err);
  process.exit(1);
});
