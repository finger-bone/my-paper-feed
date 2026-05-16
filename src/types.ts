/** A single paper entry from any source */
export interface Paper {
  id: string;
  title: string;
  authors: string[];
  /** Original abstract */
  abstract: string;
  /** AI-generated short summary (populated after LLM processing) */
  aiSummary: string;
  published: Date;
  updated: Date;
  link: string;
  pdfLink: string;
  categories: string[];
  source: "arxiv" | "huggingface";
  /** 0-10 relevance score from LLM */
  relevanceScore: number;
}

/** Configuration loaded from environment variables */
export interface Config {
  deepseekApiKey: string;
  deepseekBaseUrl: string;
  deepseekModel: string;
  /** Max papers to fetch from arXiv per query */
  arxivMaxResults: number;
  /** How many days back to look for papers */
  lookbackDays: number;
  /** Minimum relevance score (0-10) to include in RSS */
  minRelevanceScore: number;
  /** Keyword filter: auto-include threshold (default 7) */
  filterAutoInclude: number;
  /** Keyword filter: auto-exclude threshold (default 2) */
  filterAutoExclude: number;
  /** Output path for the RSS feed file */
  outputPath: string;
  /** RSS feed metadata */
  feedTitle: string;
  feedDescription: string;
  feedLink: string;
}

export function loadConfig(): Config {
  return {
    deepseekApiKey: process.env.DEEPSEEK_API_KEY || "",
    deepseekBaseUrl: process.env.DEEPSEEK_BASE_URL || "https://api.deepseek.com",
    deepseekModel: process.env.DEEPSEEK_MODEL || "deepseek-chat",
    arxivMaxResults: parseInt(process.env.ARXIV_MAX_RESULTS || "300", 10),
    lookbackDays: parseInt(process.env.LOOKBACK_DAYS || "7", 10),
    minRelevanceScore: parseInt(process.env.MIN_RELEVANCE_SCORE || "5", 10),
    filterAutoInclude: parseInt(process.env.FILTER_AUTO_INCLUDE || "6", 10),
    filterAutoExclude: parseInt(process.env.FILTER_AUTO_EXCLUDE || "1", 10),
    outputPath: process.env.OUTPUT_PATH || "feed.xml",
    feedTitle: process.env.FEED_TITLE || "LLM Acceleration Papers Weekly",
    feedDescription:
      process.env.FEED_DESCRIPTION ||
      "每周精选大模型(LLM/VLM/视频模型/世界模型)推理与训练加速相关论文",
    feedLink: process.env.FEED_LINK || "https://arxiv.org",
  };
}
