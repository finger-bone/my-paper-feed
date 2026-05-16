import { XMLParser } from "fast-xml-parser";
import type { Paper } from "../types";

const ARXIV_API = "https://export.arxiv.org/api/query";
const USER_AGENT = "PaperFeed/1.0 (mailto:example@example.com)";

/** Build the arXiv search query URL */
function buildQueryUrl(keywords: string[], maxResults: number, daysBack: number): string {
  const now = new Date();
  const past = new Date();
  past.setDate(past.getDate() - daysBack);

  const toDate = formatArxivDate(now);
  const fromDate = formatArxivDate(past);

  const categories = ["cs.CL", "cs.CV", "cs.LG", "cs.AI", "cs.MM"];
  const catQuery = categories.map((c) => `cat:${c}`).join("+OR+");

  const kwQuery = keywords
    .map((k) => `abs:${k}`)
    .join("+OR+");

  const query = `search_query=(${catQuery})+AND+(${kwQuery})`;
  const params = `${query}&sortBy=submittedDate&sortOrder=descending&max_results=${maxResults}`;

  return `${ARXIV_API}?${params}`;
}

/** Fetch papers from arXiv */
export async function fetchArxivPapers(
  maxResults: number,
  daysBack: number,
): Promise<Paper[]> {
  const keywords = [
    "efficient",
    "acceleration",
    "inference",
    "training",
    "distillation",
    "quantization",
    "pruning",
    "speculative",
    "parallelism",
    "compression",
    "low-rank",
    "attention",
    "KV cache",
    "mixture of experts",
    "MoE",
    "diffusion",
    "autoregressive",
    "generation",
  ];

  const url = buildQueryUrl(keywords, maxResults, daysBack);
  console.log(`[arxiv] Fetching: ${url}`);

  const resp = await fetch(url, {
    headers: { "User-Agent": USER_AGENT },
  });

  if (!resp.ok) {
    throw new Error(`arXiv API returned ${resp.status}: ${await resp.text()}`);
  }

  const xml = await resp.text();
  return parseArxivResponse(xml);
}

function parseArxivResponse(xml: string): Paper[] {
  const parser = new XMLParser({
    ignoreAttributes: false,
    attributeNamePrefix: "@_",
    isArray: (name) =>
      ["entry", "author", "category", "link"].includes(name),
  });

  const doc = parser.parse(xml);
  const feed = doc.feed;
  if (!feed || !feed.entry) return [];

  const entries = Array.isArray(feed.entry) ? feed.entry : [feed.entry];
  const papers: Paper[] = [];

  for (const entry of entries) {
    const id = entry.id || "";
    const arxivId = id.replace(/^https?:\/\/arxiv\.org\/abs\//, "").replace(/v\d+$/, "");

    const title = cleanHtml(entry.title || "");
    const summary = cleanHtml(entry.summary || "").replace(/\s+/g, " ").trim();

    const authors: string[] = [];
    if (entry.author) {
      for (const a of entry.author) {
        if (a.name) authors.push(a.name);
      }
    }

    const categories: string[] = [];
    if (entry.category) {
      for (const c of entry.category) {
        if (c["@_term"]) categories.push(c["@_term"]);
      }
    }

    const published = entry.published ? new Date(entry.published) : new Date();
    const updated = entry.updated ? new Date(entry.updated) : published;

    const link = `https://arxiv.org/abs/${arxivId}`;
    const pdfLink = `https://arxiv.org/pdf/${arxivId}`;

    papers.push({
      id: arxivId,
      title,
      abstract: summary,
      aiSummary: "",
      authors,
      published,
      updated,
      link,
      pdfLink,
      categories,
      source: "arxiv",
      relevanceScore: 0,
    });
  }

  console.log(`[arxiv] Parsed ${papers.length} papers`);
  return papers;
}

function cleanHtml(text: string): string {
  return text
    .replace(/<[^>]*>/g, "")
    .replace(/&amp;/g, "&")
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/&quot;/g, '"')
    .replace(/&#39;/g, "'")
    .trim();
}
