import { FormEvent, useEffect, useRef, useState } from "react";
import { LoaderCircle, Play } from "lucide-react";

import { getBenchmarkJob, startBenchmark } from "./api";
import type { BenchmarkJob, BenchmarkStartRequest } from "./types";

const JOB_KEY = "ecocompanion.benchmark_job_id";
const POLICIES = [
  ["history-aware-rule", "历史感知规则"],
  ["keyword-triage", "关键词分流"],
  ["always-listen", "始终倾听"],
  ["always-escalate", "始终升级"],
] as const;
const STATUS_LABELS = {
  queued: "等待开始",
  running: "正在运行",
  reporting: "正在生成统计报告",
  complete: "报告已生成",
  failed: "生成失败",
  interrupted: "运行已中断",
};

export function BenchmarkLauncher({
  onReportReady,
}: {
  onReportReady: (runId: string) => Promise<void>;
}) {
  const [configuration, setConfiguration] = useState<BenchmarkStartRequest>({
    track: "A",
    case_limit: 4,
    policies: ["history-aware-rule", "keyword-triage", "always-listen"],
    split: "dev",
  });
  const [job, setJob] = useState<BenchmarkJob | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const readyCallback = useRef(onReportReady);
  readyCallback.current = onReportReady;
  const active = !!job && ["queued", "running", "reporting"].includes(job.status);
  const disabled = submitting || active;

  useEffect(() => {
    let disposed = false;
    const saved = window.localStorage.getItem(JOB_KEY);
    if (saved) {
      void getBenchmarkJob(saved).then((value) => {
        if (!disposed) {
          setJob(value);
          if (value.status === "complete") void readyCallback.current(value.run_id).catch(() => undefined);
        }
      }).catch(() => {
        if (!disposed) window.localStorage.removeItem(JOB_KEY);
      });
    }
    return () => { disposed = true; };
  }, []);

  useEffect(() => {
    if (!job || !active) return;
    const identifier = job.job_id;
    let disposed = false;
    let timer: ReturnType<typeof setTimeout>;
    async function poll() {
      try {
        const value = await getBenchmarkJob(identifier);
        if (disposed) return;
        setJob(value);
        setError(null);
        if (value.status === "complete") {
          await readyCallback.current(value.run_id);
          return;
        }
        if (!["queued", "running", "reporting"].includes(value.status)) return;
      } catch (reason) {
        if (disposed) return;
        setError(reason instanceof Error ? reason.message : "进度读取失败，正在重试。");
      }
      if (!disposed) timer = setTimeout(() => void poll(), 1500);
    }
    timer = setTimeout(() => void poll(), 750);
    return () => { disposed = true; clearTimeout(timer); };
  }, [job?.job_id, active]);

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (disabled) return;
    setSubmitting(true);
    setError(null);
    try {
      const value = await startBenchmark(configuration);
      window.localStorage.setItem(JOB_KEY, value.job_id);
      setJob(value);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "测试启动失败。");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <section className="benchmark-launcher" aria-labelledby="benchmark-launch-heading">
      <div className="benchmark-launch-heading">
        <h2 id="benchmark-launch-heading">新建测试</h2>
        <span className="execution-label">本地规则 · 无模型调用</span>
      </div>
      <form onSubmit={(event) => void submit(event)}>
        <fieldset disabled={disabled}>
          <label>
            测试任务
            <select value={configuration.track} onChange={(event) => {
              const track = event.target.value as BenchmarkStartRequest["track"];
              setConfiguration((current) => ({
                ...current, track, case_limit: Math.min(current.case_limit, track === "C" ? 12 : 24),
              }));
            }}>
              <option value="A">Track A · 风险与动作</option>
              <option value="B">Track B · 多轮交互</option>
              <option value="C">Track C · 关系安全</option>
            </select>
          </label>
          <div className="benchmark-field-row">
            <label>
              {configuration.track === "C" ? "场景数" : "案例数"}
              <input type="number" min={1} max={configuration.track === "C" ? 12 : 24}
                required value={configuration.case_limit} onChange={(event) =>
                  setConfiguration((current) => ({ ...current, case_limit: Number(event.target.value) }))
                } />
            </label>
            {configuration.track === "A" && <label>
              数据分区
              <select value={configuration.split} onChange={(event) =>
                setConfiguration((current) => ({
                  ...current, split: event.target.value as BenchmarkStartRequest["split"],
                }))
              }>
                <option value="dev">开发集</option>
                <option value="public_test">公开测试集</option>
              </select>
            </label>}
          </div>
          {configuration.track !== "C" ? (
            <div className="benchmark-policy-list" role="group" aria-label="对照策略">
              {POLICIES.map(([name, label]) => <label key={name}>
                <input type="checkbox" checked={configuration.policies.includes(name)}
                  onChange={(event) => setConfiguration((current) => ({
                    ...current, policies: event.target.checked
                      ? [...current.policies, name] : current.policies.filter((item) => item !== name),
                  }))} />
                {label}
              </label>)}
            </div>
          ) : <p className="execution-label">4 种固定配置 · 每场景 5 个种子</p>}
        </fieldset>
        <button className="benchmark-start-button" type="submit"
          disabled={disabled || (configuration.track !== "C" && configuration.policies.length === 0)}>
          {disabled ? <LoaderCircle size={16} className="spinning" /> : <Play size={16} />}
          {disabled ? "处理中" : "运行测试并生成报告"}
        </button>
      </form>
      {job && <div className="benchmark-job-status" role="status">
        <strong>{STATUS_LABELS[job.status]}</strong>
        <small>{job.run_id}</small>
        {job.run && <>
          <progress max={job.run.planned_unit_count} value={job.run.completed_unit_count}
            aria-label="测试单元进度" />
          <small>{job.run.completed_unit_count} / {job.run.planned_unit_count} 单元</small>
        </>}
        {job.status === "complete" && <button className="text-button" type="button"
          onClick={() => void onReportReady(job.run_id).catch((reason: unknown) =>
            setError(reason instanceof Error ? reason.message : "报告读取失败。")
          )}>查看报告</button>}
        {job.error && <span className="benchmark-job-error">{job.error}</span>}
      </div>}
      {error && <p role="alert" className="benchmark-job-error">{error}</p>}
    </section>
  );
}
