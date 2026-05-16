import { XMLParser } from "fast-xml-parser";
import type { Paper } from "../types";

const ARXIV_API = "https://export.arxiv.org/api/query";
const USER_AGENT = "PaperFeed/1.0 (mailto:example@example.com)";

/** Retry a fetch up to `retries` times with exponential backoff */
async function fetchWithRetry(url: string, retries = 3): Promise<Response> {
  for (let attempt = 1; attempt <= retries; attempt++) {
    const resp = await fetch(url, {
      headers: { "User-Agent": USER_AGENT },
      signal: AbortSignal.timeout(30_000),
    });
    if (resp.ok) return resp;
    if (attempt < retries) {
      const delay = 2 ** attempt * 1000;
      console.warn(`[arxiv] Retry ${attempt}/${retries} after ${delay}ms (HTTP ${resp.status})`);
      await new Promise((r) => setTimeout(r, delay));
    } else {
      throw new Error(`arXiv API returned ${resp.status} after ${retries} retries`);
    }
  }
  throw new Error("unreachable");
}

/** Fetch recent papers from arXiv by category (no keyword filter — let the local filter handle it) */
export async function fetchArxivPapers(
  maxResults: number,
  _daysBack: number,
): Promise<Paper[]> {
  // Broader query: fetch latest papers from target categories without keyword filtering
  // The keyword + LLM pipeline will do the fine-grained filtering locally.
  const categories = ["cs.CL", "cs.CV", "cs.LG", "cs.AI", "cs.MM"];
  const catQuery = categories.map((c) => `cat:${c}`).join("+OR+");
  const url = `${ARXIV_API}?search_query=(${catQuery})&sortBy=submittedDate&sortOrder=descending&max_results=${maxResults}`;

  console.log(`[arxiv] Fetching latest ${maxResults} papers from [${categories.join(", ")}]`);

  const resp = await fetchWithRetry(url);
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
    // Skip error entries
    if (entry.id?.includes("/api/errors")) continue;

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
