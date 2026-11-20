import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup } from "@testing-library/react";

import { BenchmarkLauncher } from "./BenchmarkLauncher";
import * as api from "./api";
import type { BenchmarkJob } from "./types";

const JOB: BenchmarkJob = {
  job_id: "job-1",
  run_id: "web-a-test",
  status: "running",
  execution: "local_rules",
  model_calls: 0,
  error: null,
  configuration: {
    track: "A", case_limit: 4,
    policies: ["history-aware-rule", "keyword-triage", "always-listen"], split: "dev",
  },
  run: null,
};

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
  window.localStorage.clear();
});

describe("Benchmark launcher", () => {
  it("starts a real job, prevents double submission, and opens the completed report", async () => {
    const start = vi.spyOn(api, "startBenchmark").mockResolvedValue(JOB);
    vi.spyOn(api, "getBenchmarkJob").mockResolvedValue({ ...JOB, status: "complete" });
    const ready = vi.fn().mockResolvedValue(undefined);
    render(<BenchmarkLauncher onReportReady={ready} />);
    expect(screen.getByText("本地规则 · 无模型调用")).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "运行测试并生成报告" }));
    expect(start).toHaveBeenCalledWith(JOB.configuration);
    expect(window.localStorage.getItem("ecocompanion.benchmark_job_id")).toBe("job-1");
    expect(screen.getByRole("button", { name: "处理中" })).toBeDisabled();
    await waitFor(() => expect(ready).toHaveBeenCalledWith("web-a-test"), { timeout: 2500 });
    expect(screen.getByText("报告已生成")).toBeInTheDocument();
  });

  it("restores an unfinished job and exposes failure without losing its identifier", async () => {
    window.localStorage.setItem("ecocompanion.benchmark_job_id", "job-1");
    vi.spyOn(api, "getBenchmarkJob").mockResolvedValue({
      ...JOB, status: "failed", error: "运行记录已保留",
    });
    render(<BenchmarkLauncher onReportReady={vi.fn()} />);
    expect(await screen.findByText("生成失败")).toBeInTheDocument();
    expect(screen.getByText("运行记录已保留")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "运行测试并生成报告" })).toBeEnabled();
  });

  it("bounds Track C selections and displays server errors", async () => {
    vi.spyOn(api, "startBenchmark").mockRejectedValue(new Error("已有测试正在运行"));
    render(<BenchmarkLauncher onReportReady={vi.fn()} />);
    await userEvent.selectOptions(screen.getByLabelText("测试任务"), "C");
    expect(screen.getByRole("spinbutton", { name: "场景数" })).toHaveAttribute("max", "12");
    expect(screen.getByText("4 种固定配置 · 每场景 5 个种子")).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "运行测试并生成报告" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("已有测试正在运行");
  });
});
