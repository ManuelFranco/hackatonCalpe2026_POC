// Render exact HTML fixtures, preserving layout across each A/B pair.
// Requires Playwright; optionally set PLAYWRIGHT_MODULE and CHROMIUM_PATH.
import { createRequire } from 'node:module';
import { readFile, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const require = createRequire(import.meta.url);
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const root = dirname(fileURLToPath(import.meta.url));
const { cases } = JSON.parse(await readFile(resolve(root, 'ground_truth.json'), 'utf8'));
const browser = await chromium.launch({
  headless: true,
  ...(process.env.CHROMIUM_PATH ? { executablePath: process.env.CHROMIUM_PATH } : {}),
});
const report = [];
try {
  const page = await browser.newPage({ viewport: { width: 1200, height: 900 }, deviceScaleFactor: 1 });
  await page.route(/^https?:\/\//, route => route.abort());
  for (const item of cases.filter(c => c.family === 'web')) {
    await page.goto(pathToFileURL(resolve(root, item.html)).href);
    await page.evaluate(() => document.fonts.ready);
    const payload = page.locator('.payload');
    const overflow = await payload.evaluate(e => e.scrollHeight > e.clientHeight || e.scrollWidth > e.clientWidth);
    if (overflow) throw new Error(`Text clipped in ${item.id}`);
    const bounds = await payload.boundingBox();
    await page.screenshot({ path: resolve(root, item.image) });
    report.push({ id: item.id, width: 1200, height: 900, payload_bounds: bounds, clipped: false });
  }
} finally {
  await browser.close();
}
await writeFile(resolve(root, 'render_report.json'), JSON.stringify(report, null, 2) + '\n');
console.log(`Rendered ${report.length} PNGs without clipped payload text.`);
