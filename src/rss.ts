import { Feed } from "feed";
import type { Config, Paper } from "./types";
import { writeFileSync } from "node:fs";
import { join } from "node:path";

/**
 * Generate an RSS 2.0 / Atom feed from the paper list
 * and write it to the configured output path.
 */
export function generateRssFeed(papers: Paper[], config: Config): void {
  const feed = new Feed({
    title: config.feedTitle,
    description: config.feedDescription,
    id: config.feedLink,
    link: config.feedLink,
    language: "en",
    updated: papers.length > 0 ? papers[0]!.published : new Date(),
    generator: "paper-feed",
    copyright: `All rights reserved ${new Date().getFullYear()}`,
  });

  for (const paper of papers) {
    const contentParts: string[] = [];

    // AI Summary (Chinese)
    if (paper.aiSummary) {
      contentParts.push(`<h3>AI 摘要</h3><p>${escapeHtml(paper.aiSummary)}</p>`);
    }

    // Original abstract
    contentParts.push(`<h3>原始摘要</h3><p>${escapeHtml(paper.abstract)}</p>`);

    // Categories
    if (paper.categories.length > 0) {
      contentParts.push(
        `<p><strong>Categories:</strong> ${paper.categories.join(", ")}</p>`,
      );
    }

    // Relevance
    contentParts.push(
      `<p><strong>Relevance:</strong> ${paper.relevanceScore}/10</p>`,
    );

    // Links
    contentParts.push(
      `<p><a href="${escapeHtml(paper.pdfLink)}">PDF</a> | <a href="${escapeHtml(paper.link)}">arXiv</a></p>`,
    );

    feed.addItem({
      title: paper.title,
      id: paper.link,
      link: paper.link,
      description: paper.abstract.slice(0, 500),
      content: contentParts.join("\n"),
      author: paper.authors.map((name) => ({ name })),
      date: paper.published,
      published: paper.published,
      category: paper.categories.map((c: string) => ({ name: c, term: c })),
    });
  }

  // Write RSS 2.0
  const rss = feed.rss2();
  const outPath = join(process.cwd(), config.outputPath);
  writeFileSync(outPath, rss, "utf-8");
  console.log(`[rss] Feed written to ${outPath} (${papers.length} items)`);

  // Also write Atom for GitHub Pages compatibility
  const atom = feed.atom1();
  const atomPath = outPath.replace(/\.xml$/, ".atom");
  writeFileSync(atomPath, atom, "utf-8");
  console.log(`[rss] Atom feed written to ${atomPath}`);
}

function escapeHtml(text: string): string {
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}
