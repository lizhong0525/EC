/* End-to-end checks against an explicitly started local research preview. */
const { chromium } = require("playwright");
const fs = require("node:fs/promises");
const path = require("node:path");

async function main() {
  const output = path.resolve(process.argv[2]);
  await fs.mkdir(output, { recursive: false });
  const browser = await chromium.launch({
    headless: true, channel: process.env.PLAYWRIGHT_CHANNEL,
  });
  const context = await browser.newContext({
    baseURL: "http://127.0.0.1:8002",
    viewport: { width: 1440, height: 1000 },
    acceptDownloads: true,
  });
  const page = await context.newPage();
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  const runs = [];
  try {
    await page.goto("http://127.0.0.1:8002/", { waitUntil: "networkidle" });
    await page.getByRole("button", { name: "Benchmark", exact: true }).click();
    for (const track of ["A", "B", "C"]) {
      await page.getByLabel("测试任务").selectOption(track);
      await page.getByRole("spinbutton").fill("1");
      await page.getByRole("button", { name: "运行测试并生成报告" }).click();
      await page.getByRole("button", { name: "处理中", exact: true }).waitFor();
      await page.locator(".benchmark-job-status strong").filter({
        hasText: /等待开始|正在运行|正在生成统计报告/,
      }).waitFor();
      if (track === "A") {
        await page.reload({ waitUntil: "networkidle" });
        await page.getByRole("button", { name: "Benchmark", exact: true }).click();
      }
      await page.getByText("报告已生成", { exact: true }).waitFor({ timeout: 180000 });
      const jobId = await page.evaluate(() =>
        localStorage.getItem("ecocompanion.benchmark_job_id"),
      );
      const response = await page.request.get(`/api/v1/research/benchmark/jobs/${jobId}`);
      if (!response.ok()) throw new Error("job query failed");
      const job = await response.json();
      if (job.status !== "complete" || job.model_calls !== 0) {
        throw new Error(`unexpected job state: ${JSON.stringify(job)}`);
      }
      await page.locator(".report-heading h2").filter({ hasText: job.run_id }).waitFor();
      if (!(await page.locator(".report-table tbody tr").count())) {
        throw new Error("report table is empty");
      }
      for (const format of ["md", "json"]) {
        const downloadPromise = page.waitForEvent("download");
        await page.getByRole("link", {
          name: format === "md" ? "Markdown 报告" : "JSON 数据",
        }).click();
        const download = await downloadPromise;
        const target = path.join(output, `${track}-report.${format}`);
        await download.saveAs(target);
        if ((await fs.stat(target)).size < 50) throw new Error("empty download");
        if (format === "json") {
          const report = JSON.parse(await fs.readFile(target, "utf8"));
          if (report.run_id !== job.run_id || !report.complete) {
            throw new Error("download does not match selected report");
          }
        }
      }
      runs.push({
        track, job_id: jobId, run_id: job.run_id,
        completed_units: job.run.completed_unit_count,
        planned_units: job.run.planned_unit_count,
      });
      console.log(`${track}: generated, displayed, and downloaded ${job.run_id}`);
    }
    const overflow = {};
    for (const [name, viewport] of [
      ["desktop", { width: 1440, height: 1000 }],
      ["mobile", { width: 390, height: 844 }],
    ]) {
      await page.setViewportSize(viewport);
      await page.screenshot({ path: path.join(output, `${name}.png`), fullPage: true });
      overflow[name] = await page.evaluate(() =>
        document.documentElement.scrollWidth > innerWidth,
      );
      if (overflow[name]) throw new Error(`${name}: horizontal page overflow`);
    }
    if (errors.length) throw new Error(`browser errors: ${errors.join("; ")}`);
    await fs.writeFile(path.join(output, "verification.json"), JSON.stringify({
      runs, refresh_recovery: true, downloads_verified: 6,
      page_errors: errors, horizontal_overflow: overflow,
    }, null, 2));
  } finally {
    await context.close();
    await browser.close();
  }
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
