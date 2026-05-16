import type { Paper } from "../types";

const HF_DAILY_PAPERS = "https://huggingface.co/api/daily_papers";

/** Fetch daily papers from HuggingFace for the last N days */
export async function fetchHuggingFacePapers(
  _maxResults: number,
  daysBack: number,
): Promise<Paper[]> {
  const allPapers: Paper[] = [];

  // Fetch papers for each day
  for (let i = 0; i < daysBack; i++) {
    const date = new Date();
    date.setDate(date.getDate() - i);
    const dateStr = date.toISOString().slice(0, 10);

    try {
      const papers = await fetchDailyPapers(dateStr);
      allPapers.push(...papers);
      console.log(`[huggingface] ${dateStr}: ${papers.length} papers`);
    } catch (err) {
      console.warn(`[huggingface] Failed to fetch ${dateStr}:`, err);
    }
  }

  // Deduplicate by arXiv ID
  const seen = new Set<string>();
  const unique: Paper[] = [];
  for (const p of allPapers) {
    if (!seen.has(p.id)) {
      seen.add(p.id);
      unique.push(p);
    }
  }

  console.log(`[huggingface] Total unique papers: ${unique.length}`);
  return unique;
}

async function fetchDailyPapers(date: string): Promise<Paper[]> {
  const url = `${HF_DAILY_PAPERS}?date=${date}&limit=50`;
  const resp = await fetch(url, {
    headers: { Accept: "application/json" },
    signal: AbortSignal.timeout(15000),
  });

  if (!resp.ok) {
    // 404 means no papers for that day
    if (resp.status === 404) return [];
    throw new Error(`HuggingFace API returned ${resp.status}`);
  }

  const data = await resp.json();
  return parseHfResponse(data);
}

function parseHfResponse(data: unknown): Paper[] {
  if (!Array.isArray(data)) return [];

  const papers: Paper[] = [];

  for (const item of data) {
    const paper = item.paper;
    if (!paper) continue;

    const arxivId = extractArxivId(paper.id || "");
    if (!arxivId) continue;

    const title = paper.title || "";
    const summary = paper.summary || "";

    const authors: string[] = [];
    if (paper.authors) {
      const authorList = Array.isArray(paper.authors)
        ? paper.authors
        : [paper.authors];
      for (const a of authorList) {
        if (typeof a === "string") authors.push(a);
        else if (a?.name) authors.push(a.name);
      }
    }

    const categories: string[] = [];
    if (paper.tags) {
      const tags = Array.isArray(paper.tags) ? paper.tags : [paper.tags];
      for (const t of tags) {
        if (typeof t === "string") categories.push(t);
      }
    }

    const published = paper.publishedAt
      ? new Date(paper.publishedAt)
      : paper.publicationDate
        ? new Date(paper.publicationDate)
        : new Date();
    const updated = paper.updatedAt ? new Date(paper.updatedAt) : published;

    papers.push({
      id: arxivId,
      title,
      abstract: summary,
      aiSummary: "",
      authors,
      published,
      updated,
      link: `https://arxiv.org/abs/${arxivId}`,
      pdfLink: `https://arxiv.org/pdf/${arxivId}`,
      categories,
      source: "huggingface",
      relevanceScore: 0,
    });
  }

  return papers;
}

function extractArxivId(id: string): string | null {
  // HF paper IDs can be like "arxiv:2501.00001" or just "2501.00001"
  const match = id.match(/(?:arxiv:)?(\d{4}\.\d{4,5})(?:v\d+)?/i);
  return (match?.[1] as string) || null;
}
