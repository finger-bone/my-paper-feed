import type { Paper } from "../types";

const HF_DAILY_PAPERS = "https://huggingface.co/api/daily_papers";

/** Retry a fetch up to `retries` times */
async function fetchWithRetry(url: string, retries = 3): Promise<Response> {
  for (let attempt = 1; attempt <= retries; attempt++) {
    try {
      const resp = await fetch(url, {
        headers: { Accept: "application/json" },
        signal: AbortSignal.timeout(15_000),
      });
      // 404 means no papers for that day — not an error
      if (resp.status === 404) return resp;
      if (resp.ok) return resp;
      if (attempt < retries) {
        const delay = 2 ** attempt * 1000;
        console.warn(`[huggingface] Retry ${attempt}/${retries} after ${delay}ms (HTTP ${resp.status})`);
        await new Promise((r) => setTimeout(r, delay));
      } else {
        throw new Error(`HuggingFace API returned ${resp.status}`);
      }
    } catch (err) {
      if (attempt < retries && !(err instanceof Error && err.message.includes("returned"))) {
        const delay = 2 ** attempt * 1000;
        console.warn(`[huggingface] Retry ${attempt}/${retries} after ${delay}ms:`, err);
        await new Promise((r) => setTimeout(r, delay));
      } else {
        throw err;
      }
    }
  }
  throw new Error("unreachable");
}

/** Fetch daily papers from HuggingFace for the last N days (in parallel) */
export async function fetchHuggingFacePapers(
  _maxResults: number,
  daysBack: number,
): Promise<Paper[]> {
  const dates: string[] = [];
  for (let i = 0; i < daysBack; i++) {
    const date = new Date();
    date.setDate(date.getDate() - i);
    dates.push(date.toISOString().slice(0, 10));
  }

  // Fetch all days in parallel
  const results = await Promise.allSettled(
    dates.map(async (dateStr) => {
      const papers = await fetchDailyPapers(dateStr);
      return { dateStr, papers };
    }),
  );

  const allPapers: Paper[] = [];
  for (const result of results) {
    if (result.status === "fulfilled") {
      const { dateStr, papers } = result.value;
      if (papers.length > 0) {
        console.log(`[huggingface] ${dateStr}: ${papers.length} papers`);
      }
      allPapers.push(...papers);
    } else {
      console.warn(`[huggingface] A day fetch failed:`, result.reason);
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
  const resp = await fetchWithRetry(url);

  if (resp.status === 404) return [];
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
  const match = id.match(/(?:arxiv:)?(\d{4}\.\d{4,5})(?:v\d+)?/i);
  return (match?.[1] as string) || null;
}
