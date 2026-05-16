import { loadConfig, type Config, type Paper } from "./types";
import { fetchArxivPapers } from "./sources/arxiv";
import { fetchHuggingFacePapers } from "./sources/huggingface";
import { keywordFilter, analyzeUncertainPapers, generateSummariesForDecided, applyJudgments } from "./llm";
import { generateRssFeed } from "./rss";
import { loadCache, saveCache, updateCache } from "./cache";

async function main() {
  console.log("=".repeat(60));
  console.log("Paper Feed — LLM Acceleration Papers Curator");
  console.log("=".repeat(60));

  const config = loadConfig();
  validateConfig(config);

  // Phase 1: Fetch papers from all sources
  console.log("\n[1/5] Fetching papers...");
  const allPapers = await fetchAllSources(config);
  console.log(`  Total papers fetched: ${allPapers.length}`);

  if (allPapers.length === 0) {
    console.log("  No papers found. Exiting.");
    return;
  }

  // Phase 2: Keyword pre-filter (lenient — prefers LLM over premature exclusion)
  console.log("\n[2/5] Keyword pre-filtering...");
  const { decided, uncertain } = keywordFilter(allPapers, {
    autoInclude: config.filterAutoInclude,
    autoExclude: config.filterAutoExclude,
  });
  const allJudgments = new Map(decided);

  // Load cache
  const cache = loadCache();

  // Phase 3: LLM analysis for uncertain papers
  console.log("\n[3/5] LLM analysis (uncertain papers only)...");
  const llmJudgments = await analyzeUncertainPapers(uncertain, config, cache);

  // Merge LLM results and update cache
  updateCache(cache, llmJudgments);
  for (const [id, j] of llmJudgments) {
    allJudgments.set(id, j);
  }

  // Phase 3b: Generate Chinese summaries for keyword-decided papers
  console.log("\n[3b/5] Generating Chinese summaries for keyword-decided papers...");
  const summaryJudgments = await generateSummariesForDecided(allPapers, config, cache);
  updateCache(cache, summaryJudgments);
  for (const [id, j] of summaryJudgments) {
    allJudgments.set(id, j);
  }

  // Persist updated cache
  saveCache(cache);

  // Phase 4: Filter and sort
  console.log("\n[4/5] Filtering by relevance...");
  const relevantPapers = applyJudgments(allPapers, allJudgments, config.minRelevanceScore);

  if (relevantPapers.length === 0) {
    console.log("  No relevant papers found. Generating empty feed.");
  } else {
    console.log(`  Relevant papers (score >= ${config.minRelevanceScore}):`);
    for (const p of relevantPapers) {
      console.log(`  [${p.relevanceScore}/10] ${p.title}`);
      if (p.aiSummary) {
        console.log(`         ${p.aiSummary.slice(0, 120)}...`);
      }
      console.log(`         ${p.link}`);
      console.log();
    }
  }

  // Phase 5: Generate RSS
  console.log("\n[5/5] Generating RSS feed...");
  generateRssFeed(relevantPapers, config);

  // Summary
  const autoIncluded = [...decided.values()].filter((j) => j.relevanceScore >= config.filterAutoInclude).length;
  const autoExcluded = [...decided.values()].filter((j) => j.relevanceScore <= config.filterAutoExclude).length;

  console.log("\n" + "=".repeat(60));
  console.log("Done!");
  console.log(`  Papers fetched:      ${allPapers.length}`);
  console.log(`  Keyword auto-include: ${autoIncluded}`);
  console.log(`  Keyword auto-exclude: ${autoExcluded}`);
  console.log(`  LLM analyzed:        ${llmJudgments.size}`);
  console.log(`  Summaries generated:  ${summaryJudgments.size}`);
  console.log(`  Relevant (in RSS):   ${relevantPapers.length}`);
  if (relevantPapers.length > 0) {
    console.log(
      `  Avg relevance:       ${(
        relevantPapers.reduce((s, p) => s + p.relevanceScore, 0) /
        relevantPapers.length
      ).toFixed(1)}/10`,
    );
  }
  console.log(`  Cache entries:       ${cache.size}`);
  console.log(`  Output:              ${config.outputPath}`);
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
      console.log(`  ${sourceName}: ${result.value.length} papers`);
    } else {
      console.error(`  ${sourceName} failed:`, result.reason);
    }
  }

  return papers;
}

function validateConfig(config: Config): void {
  if (!config.deepseekApiKey) {
    console.error(
      "DEEPSEEK_API_KEY environment variable is required.\n" +
        "   Set it in GitHub Secrets or your local .env file.",
    );
    process.exit(1);
  }

  if (config.lookbackDays < 1 || config.lookbackDays > 30) {
    console.warn("  LOOKBACK_DAYS should be between 1 and 30. Using 7.");
    config.lookbackDays = 7;
  }

  if (config.minRelevanceScore < 0 || config.minRelevanceScore > 10) {
    console.warn("  MIN_RELEVANCE_SCORE should be 0-10. Using 5.");
    config.minRelevanceScore = 5;
  }

  if (config.filterAutoInclude < 1 || config.filterAutoInclude > 10) {
    console.warn("  FILTER_AUTO_INCLUDE should be 1-10. Using 8.");
    config.filterAutoInclude = 7;
  }

  if (config.filterAutoExclude < 0 || config.filterAutoExclude > 5) {
    console.warn("  FILTER_AUTO_EXCLUDE should be 0-5. Using 1.");
    config.filterAutoExclude = 3;
  }
}

main().catch((err) => {
  console.error("Fatal error:", err);
  process.exit(1);
});
