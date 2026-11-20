import type {
  BenchmarkReport,
  BenchmarkJob,
  BenchmarkStartRequest,
  BenchmarkRunCatalog,
  MemoryRecord,
  ModelSnapshot,
  ProactiveMessage,
  SessionView,
  StoredMessage,
  TraceRecord,
  TurnResult,
} from "./types";

const JSON_HEADERS = {
  "Content-Type": "application/json",
};

async function requireOk(response: Response): Promise<Response> {
  if (response.ok) {
    return response;
  }
  let detail = `${response.status} ${response.statusText}`;
  try {
    const body = (await response.json()) as { detail?: string };
    detail = body.detail ?? detail;
  } catch {
    // Keep the HTTP status when the body is not JSON.
  }
  throw new Error(detail);
}

export async function createSession(): Promise<SessionView> {
  const response = await requireOk(
    await fetch("/api/v1/sessions", {
      method: "POST",
      headers: JSON_HEADERS,
      body: JSON.stringify({
        participant_type: "adult_research_tester",
        memory_enabled: true,
        proactive_enabled: false,
      }),
    }),
  );
  return (await response.json()) as SessionView;
}

export async function getModelSnapshot(): Promise<ModelSnapshot> {
  const response = await requireOk(await fetch("/api/v1/models"));
  return (await response.json()) as ModelSnapshot;
}

export async function getSession(sessionId: string): Promise<SessionView> {
  const response = await requireOk(
    await fetch(`/api/v1/sessions/${sessionId}`),
  );
  return (await response.json()) as SessionView;
}

export async function getMessages(sessionId: string): Promise<StoredMessage[]> {
  const response = await requireOk(
    await fetch(`/api/v1/sessions/${sessionId}/messages`),
  );
  return (await response.json()) as StoredMessage[];
}

export async function sendTurn(
  sessionId: string,
  text: string,
  onToken: (token: string) => void,
): Promise<TurnResult> {
  const response = await requireOk(
    await fetch(`/api/v1/sessions/${sessionId}/turn`, {
      method: "POST",
      headers: {
        ...JSON_HEADERS,
        Accept: "text/event-stream, application/json",
      },
      body: JSON.stringify({ text }),
    }),
  );

  const contentType = response.headers.get("content-type") ?? "";
  if (!contentType.includes("text/event-stream")) {
    return (await response.json()) as TurnResult;
  }

  if (!response.body) {
    throw new Error("流式响应没有返回可读取的内容。");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let finalResult: TurnResult | null = null;

  while (true) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value, { stream: !done });
    const frames = buffer.split("\n\n");
    buffer = frames.pop() ?? "";

    for (const frame of frames) {
      const eventLine = frame
        .split("\n")
        .find((line) => line.startsWith("event:"));
      const dataLines = frame
        .split("\n")
        .filter((line) => line.startsWith("data:"))
        .map((line) => line.slice(5).trim());
      if (dataLines.length === 0) {
        continue;
      }
      const event = eventLine?.slice(6).trim() ?? "message";
      const payload = JSON.parse(dataLines.join("\n")) as {
        token?: string;
        result?: TurnResult;
        detail?: string;
      };
      if (event === "token" && payload.token) {
        onToken(payload.token);
      }
      if (event === "done" && payload.result) {
        finalResult = payload.result;
      }
      if (event === "error") {
        throw new Error(payload.detail ?? "流式对话失败。");
      }
    }

    if (done) {
      break;
    }
  }

  if (!finalResult) {
    throw new Error("流式响应结束但没有最终结果。");
  }
  return finalResult;
}

export async function getMemories(sessionId: string): Promise<MemoryRecord[]> {
  const response = await requireOk(
    await fetch(`/api/v1/sessions/${sessionId}/memory`),
  );
  return (await response.json()) as MemoryRecord[];
}

export async function deleteMemory(
  sessionId: string,
  memoryId: string,
): Promise<void> {
  await requireOk(
    await fetch(`/api/v1/sessions/${sessionId}/memory/${memoryId}`, {
      method: "DELETE",
    }),
  );
}

export async function setProactivity(
  sessionId: string,
  enabled: boolean,
): Promise<SessionView> {
  const response = await requireOk(
    await fetch(`/api/v1/sessions/${sessionId}/proactivity`, {
      method: "PUT",
      headers: JSON_HEADERS,
      body: JSON.stringify({ enabled }),
    }),
  );
  return (await response.json()) as SessionView;
}

export async function claimProactiveMessage(
  sessionId: string,
): Promise<ProactiveMessage> {
  const response = await requireOk(
    await fetch(`/api/v1/sessions/${sessionId}/proactivity/claim`, {
      method: "POST",
    }),
  );
  return (await response.json()) as ProactiveMessage;
}

export async function getTrace(sessionId: string): Promise<TraceRecord[]> {
  const response = await requireOk(
    await fetch(`/api/v1/research/sessions/${sessionId}/trace`),
  );
  return (await response.json()) as TraceRecord[];
}

export async function getBenchmarkRuns(): Promise<BenchmarkRunCatalog> {
  const response = await requireOk(
    await fetch("/api/v1/research/benchmark/runs"),
  );
  return (await response.json()) as BenchmarkRunCatalog;
}

export async function getBenchmarkReport(
  runId: string,
): Promise<BenchmarkReport> {
  const response = await requireOk(
    await fetch(
      `/api/v1/research/benchmark/runs/${encodeURIComponent(runId)}/report`,
    ),
  );
  return (await response.json()) as BenchmarkReport;
}

export async function startBenchmark(
  configuration: BenchmarkStartRequest,
): Promise<BenchmarkJob> {
  const response = await requireOk(
    await fetch("/api/v1/research/benchmark/jobs", {
      method: "POST",
      headers: JSON_HEADERS,
      body: JSON.stringify(configuration),
    }),
  );
  return (await response.json()) as BenchmarkJob;
}

export async function getBenchmarkJob(jobId: string): Promise<BenchmarkJob> {
  const response = await requireOk(
    await fetch(`/api/v1/research/benchmark/jobs/${encodeURIComponent(jobId)}`),
  );
  return (await response.json()) as BenchmarkJob;
}

export async function generateBenchmarkReport(runId: string): Promise<BenchmarkReport> {
  const response = await requireOk(
    await fetch(
      `/api/v1/research/benchmark/runs/${encodeURIComponent(runId)}/report`,
      { method: "POST" },
    ),
  );
  return (await response.json()) as BenchmarkReport;
}
