import {
  useCallback,
  ChangeEvent,
  DragEvent,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { DecisionRecord, sampleDecisions } from "./sample-data";

type Filter = "all" | "selected" | "silent" | "suppressed";

const number = (value: unknown, fallback = 0) =>
  typeof value === "number" ? value : fallback;
const text = (value: unknown, fallback = "—") =>
  typeof value === "string" && value.length ? value : fallback;
const list = (value: unknown) => (Array.isArray(value) ? value : []);

function candidateScore(candidate: Record<string, unknown>) {
  return Math.max(
    number(candidate.contextual_reranker_score, -1),
    number(candidate.reaction_reranker_score, -1),
  );
}

function formatTime(value?: string) {
  if (!value) return "時間未記錄";
  const date = new Date(value);
  return Number.isNaN(date.getTime())
    ? value
    : new Intl.DateTimeFormat("zh-TW", {
        month: "2-digit",
        day: "2-digit",
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit",
        hour12: false,
      }).format(date);
}

function formatDuration(value: number) {
  if (value < 1000) return `${Math.round(value)} ms`;
  return `${(value / 1000).toFixed(1)} s`;
}

function statusOf(record: DecisionRecord) {
  return text(record.result?.status, "unknown");
}

function deliveryOf(record: DecisionRecord) {
  return text(record.result?.delivery, "unknown");
}

function parseJsonl(source: string): DecisionRecord[] {
  const records = source
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line, index) => {
      try {
        return JSON.parse(line) as DecisionRecord;
      } catch {
        throw new Error(`第 ${index + 1} 行不是有效的 JSON`);
      }
    });
  if (!records.length) throw new Error("檔案內沒有決策紀錄");
  return records.reverse();
}

function Stat({
  label,
  value,
  detail,
  tone,
}: {
  label: string;
  value: string;
  detail: string;
  tone?: "orange" | "green";
}) {
  return (
    <div className={`stat-card ${tone ? `stat-${tone}` : ""}`}>
      <span>{label}</span>
      <strong>{value}</strong>
      <small>{detail}</small>
    </div>
  );
}

function Tag({ children, tone = "neutral" }: {
  children: React.ReactNode;
  tone?: "neutral" | "green" | "orange" | "red";
}) {
  return <span className={`tag tag-${tone}`}>{children}</span>;
}

function Stage({
  index,
  title,
  subtitle,
  duration,
  active = true,
}: {
  index: string;
  title: string;
  subtitle: string;
  duration?: number;
  active?: boolean;
}) {
  return (
    <div className={`stage ${active ? "" : "stage-muted"}`}>
      <div className="stage-index">{index}</div>
      <div className="stage-copy">
        <strong>{title}</strong>
        <span>{subtitle}</span>
      </div>
      {duration !== undefined && (
        <span className="stage-time">{formatDuration(duration)}</span>
      )}
    </div>
  );
}

export function DecisionDashboard() {
  const [records, setRecords] = useState<DecisionRecord[]>(sampleDecisions);
  const [selectedId, setSelectedId] = useState(
    sampleDecisions[0].selection_id ?? "",
  );
  const [filter, setFilter] = useState<Filter>("all");
  const [query, setQuery] = useState("");
  const [datasetName, setDatasetName] = useState("正在連線 CT108…");
  const [importError, setImportError] = useState("");
  const [lastUpdated, setLastUpdated] = useState("");
  const [refreshing, setRefreshing] = useState(true);
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const refreshLiveData = useCallback(async () => {
    setRefreshing(true);
    try {
      const response = await fetch("/api/decisions?limit=300", {
        cache: "no-store",
      });
      if (!response.ok) {
        throw new Error(`API 回傳 ${response.status}`);
      }
      const payload = (await response.json()) as {
        records?: DecisionRecord[];
        source?: string;
      };
      const nextRecords = payload.records ?? [];
      if (!nextRecords.length) {
        throw new Error("目前沒有決策紀錄");
      }
      setRecords(nextRecords);
      setSelectedId((current) =>
        nextRecords.some((record) => record.selection_id === current)
          ? current
          : (nextRecords[0]?.selection_id ?? ""),
      );
      setDatasetName(payload.source ?? "CT108 · Live");
      setLastUpdated(
        new Intl.DateTimeFormat("zh-TW", {
          hour: "2-digit",
          minute: "2-digit",
          second: "2-digit",
          hour12: false,
        }).format(new Date()),
      );
      setImportError("");
    } catch (error) {
      setDatasetName("示範資料 · API 未連線");
      setImportError(
        error instanceof Error
          ? `無法讀取本機決策 API：${error.message}`
          : "無法讀取本機決策 API",
      );
    } finally {
      setRefreshing(false);
    }
  }, []);

  useEffect(() => {
    void refreshLiveData();
    const timer = window.setInterval(() => {
      void refreshLiveData();
    }, 30_000);
    return () => window.clearInterval(timer);
  }, [refreshLiveData]);

  const filtered = useMemo(() => {
    const needle = query.trim().toLowerCase();
    return records.filter((record) => {
      const status = statusOf(record);
      const delivery = deliveryOf(record);
      const filterMatches =
        filter === "all" ||
        (filter === "selected" && status === "selected") ||
        (filter === "silent" && status === "stayed_silent") ||
        (filter === "suppressed" && delivery === "suppressed_off");
      const searchMatches =
        !needle ||
        `${record.query ?? ""}\n${record.conversation ?? ""}\n${
          record.result?.text ?? ""
        }`
          .toLowerCase()
          .includes(needle);
      return filterMatches && searchMatches;
    });
  }, [records, filter, query]);

  const selected =
    records.find((record) => record.selection_id === selectedId) ??
    filtered[0] ??
    records[0];

  const totals = useMemo(() => {
    const selectedCount = records.filter(
      (record) => statusOf(record) === "selected",
    ).length;
    const silentCount = records.filter(
      (record) => statusOf(record) === "stayed_silent",
    ).length;
    const suppressedCount = records.filter(
      (record) => deliveryOf(record) === "suppressed_off",
    ).length;
    const completeTimes = records
      .map((record) => number(record.timings_ms?.total))
      .filter((value) => value > 0);
    return {
      selectedCount,
      silentCount,
      suppressedCount,
      average:
        completeTimes.reduce((sum, value) => sum + value, 0) /
        Math.max(1, completeTimes.length),
    };
  }, [records]);

  async function loadFile(file?: File) {
    if (!file) return;
    try {
      const nextRecords = parseJsonl(await file.text());
      setRecords(nextRecords);
      setSelectedId(nextRecords[0]?.selection_id ?? "");
      setDatasetName(file.name);
      setLastUpdated("手動匯入");
      setImportError("");
    } catch (error) {
      setImportError(error instanceof Error ? error.message : "無法讀取檔案");
    }
  }

  function handleDrop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    setDragging(false);
    void loadFile(event.dataTransfer.files[0]);
  }

  if (!selected) return null;

  const planner = selected.planner ?? {};
  const rawPlanner = (planner.raw ?? {}) as Record<string, unknown>;
  const adjustments = list(planner.adjustments);
  const candidates = selected.candidates ?? [];
  const result = selected.result ?? {};
  const context = selected.context ?? {};
  const policy = selected.policy ?? {};
  const retrieval = selected.retrieval ?? {};
  const timings = selected.timings_ms ?? {};
  const baseline = (selected.cross_encoder?.baseline ?? {}) as Record<
    string,
    unknown
  >;
  const selectedSegment = result.segment_id;
  const total = number(timings.total);
  const resultStatus = statusOf(selected);
  const resultDelivery = deliveryOf(selected);
  const maxScore = Math.max(
    0.01,
    ...candidates.map((candidate) => candidateScore(candidate)),
  );

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark">M</span>
          <div>
            <strong>Mortis</strong>
            <span>Decision Observatory</span>
          </div>
        </div>
        <div className="topbar-meta">
          <span className="dataset-pill">
            <i className={refreshing ? "pulse" : ""} />
            {datasetName}
          </span>
          <button
            className="refresh-button"
            onClick={() => void refreshLiveData()}
            disabled={refreshing}
            title={lastUpdated ? `上次更新 ${lastUpdated}` : "重新整理"}
          >
            {refreshing ? "更新中…" : "重新整理"}
          </button>
          <input
            ref={inputRef}
            className="visually-hidden"
            type="file"
            accept=".jsonl,.json,application/json"
            onChange={(event: ChangeEvent<HTMLInputElement>) =>
              void loadFile(event.target.files?.[0])
            }
          />
          <button
            className="import-button"
            onClick={() => inputRef.current?.click()}
          >
            匯入 decisions.jsonl
          </button>
        </div>
      </header>

      <section className="hero">
        <div>
          <div className="eyebrow">
            <span>DECISION TRACE</span>
            <i />
            <span>LOCAL-FIRST</span>
          </div>
          <h1>每一張圖，<em>為什麼是它？</em></h1>
          <p>
            從群聊語境一路追到圖片傳送，把模型規劃、候選召回、重排分數與門檻決策放在同一條證據鏈上。
          </p>
        </div>
        <div
          className={`drop-zone ${dragging ? "drop-active" : ""}`}
          onDragOver={(event) => {
            event.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={handleDrop}
          onClick={() => inputRef.current?.click()}
          role="button"
          tabIndex={0}
          onKeyDown={(event) => {
            if (event.key === "Enter" || event.key === " ") {
              inputRef.current?.click();
            }
          }}
        >
          <span>↥</span>
          <div>
            <strong>拖放真實決策紀錄</strong>
            <small>只在這個瀏覽器解析，不會上傳對話內容</small>
          </div>
        </div>
      </section>

      {importError && <div className="error-banner">{importError}</div>}

      <section className="stats">
        <Stat
          label="決策紀錄"
          value={records.length.toLocaleString("zh-TW")}
          detail={`目前資料集 · ${datasetName}`}
        />
        <Stat
          label="選出圖片"
          value={`${Math.round((totals.selectedCount / records.length) * 100)}%`}
          detail={`${totals.selectedCount} 筆 selected`}
          tone="green"
        />
        <Stat
          label="保持沉默"
          value={`${Math.round((totals.silentCount / records.length) * 100)}%`}
          detail={`${totals.silentCount} 筆 stayed_silent`}
        />
        <Stat
          label="平均決策時間"
          value={formatDuration(totals.average)}
          detail={`${totals.suppressedCount} 筆 Off 未傳送`}
          tone="orange"
        />
      </section>

      <section className="workspace">
        <aside className="decision-list">
          <div className="list-head">
            <div>
              <span>MESSAGE STREAM</span>
              <strong>{filtered.length} 筆</strong>
            </div>
            <label className="search">
              <span>⌕</span>
              <input
                value={query}
                onChange={(event) => setQuery(event.target.value)}
                placeholder="搜尋訊息或字幕"
              />
            </label>
            <div className="filters" aria-label="決策篩選">
              {([
                ["all", "全部"],
                ["selected", "有選圖"],
                ["silent", "沉默"],
                ["suppressed", "Off"],
              ] as Array<[Filter, string]>).map(([value, label]) => (
                <button
                  key={value}
                  className={filter === value ? "active" : ""}
                  onClick={() => setFilter(value)}
                >
                  {label}
                </button>
              ))}
            </div>
          </div>
          <div className="records">
            {filtered.map((record) => {
              const status = statusOf(record);
              const delivery = deliveryOf(record);
              const isActive = record.selection_id === selected.selection_id;
              return (
                <button
                  className={`record ${isActive ? "record-active" : ""}`}
                  key={record.selection_id}
                  onClick={() => setSelectedId(record.selection_id ?? "")}
                >
                  <div className="record-top">
                    <span>{formatTime(record.timestamp)}</span>
                    <i
                      className={
                        status === "selected" ? "dot-selected" : "dot-silent"
                      }
                    />
                  </div>
                  <strong>{record.query || "未記錄訊息"}</strong>
                  <div className="record-bottom">
                    <span>{text(record.context?.author_name, "未知使用者")}</span>
                    <Tag
                      tone={
                        delivery === "suppressed_off"
                          ? "orange"
                          : status === "selected"
                            ? "green"
                            : "neutral"
                      }
                    >
                      {delivery === "suppressed_off"
                        ? "OFF"
                        : status === "selected"
                          ? "SELECTED"
                          : "SILENT"}
                    </Tag>
                  </div>
                </button>
              );
            })}
            {!filtered.length && (
              <div className="empty">目前篩選條件沒有符合的紀錄。</div>
            )}
          </div>
        </aside>

        <article className="decision-detail">
          <div className="detail-head">
            <div>
              <div className="eyebrow">
                <span>SELECTION</span>
                <code>{selected.selection_id ?? "unknown"}</code>
              </div>
              <h2>{selected.query || "未記錄訊息"}</h2>
            </div>
            <div className="detail-tags">
              <Tag tone={resultStatus === "selected" ? "green" : "neutral"}>
                {resultStatus}
              </Tag>
              <Tag tone={resultDelivery === "suppressed_off" ? "orange" : "green"}>
                {resultDelivery}
              </Tag>
            </div>
          </div>

          <div className="context-grid">
            <section className="panel conversation-panel">
              <div className="panel-title">
                <span>01</span>
                <div>
                  <strong>群聊語境</strong>
                  <small>模型實際收到的最近對話</small>
                </div>
              </div>
              <div className="conversation">
                {text(selected.conversation, "沒有前文")
                  .split("\n")
                  .map((line, index) => (
                    <p key={`${line}-${index}`}>{line}</p>
                  ))}
                <div className="latest-message">
                  <span>{text(context.author_name, "LATEST")}</span>
                  <strong>{selected.query}</strong>
                </div>
              </div>
            </section>

            <section className="panel outcome-panel">
              <div className="panel-title">
                <span>07</span>
                <div>
                  <strong>最後結果</strong>
                  <small>實際選圖與傳送狀態</small>
                </div>
              </div>
              <div className="outcome">
                <span className="outcome-label">
                  {resultStatus === "selected" ? "SELECTED CAPTION" : "NO IMAGE"}
                </span>
                <blockquote>
                  {text(
                    result.text,
                    resultStatus === "stayed_silent" ? "保持沉默" : "未記錄字幕",
                  )}
                </blockquote>
                {typeof result.image_url === "string" && (
                  <img
                    className="selected-image"
                    src={result.image_url}
                    alt={text(result.text, "選中的梗圖")}
                  />
                )}
                <div className="outcome-meta">
                  <span>segment {text(result.segment_id, "—")}</span>
                  <span>{text(policy.gate_reason, "未記錄門檻")}</span>
                </div>
              </div>
            </section>
          </div>

          <section className="trace-panel">
            <div className="section-heading">
              <div>
                <span>END-TO-END TRACE</span>
                <h3>決策流程</h3>
              </div>
              <strong>{formatDuration(total)}</strong>
            </div>
            <div className="stages">
              <Stage
                index="01"
                title="接收訊息"
                subtitle={`${text(context.configured_mode, "—")} → ${text(
                  context.evaluated_mode,
                  "—",
                )}`}
                duration={number(timings.queue)}
              />
              <Stage
                index="02"
                title="Planner"
                subtitle={`${text(planner.action, "未執行")} · ${text(
                  planner.meme_role,
                  "—",
                )}`}
                duration={number(timings.planner)}
                active={Boolean(planner.action)}
              />
              <Stage
                index="03"
                title="Embedding"
                subtitle={`${list(retrieval.queries).length} 個語意查詢`}
                duration={number(timings.embedding)}
                active={Boolean(retrieval.mode)}
              />
              <Stage
                index="04"
                title="候選召回"
                subtitle={`${number(retrieval.pool_size)} 張混合候選`}
                duration={number(timings.retrieval)}
                active={Boolean(retrieval.mode)}
              />
              <Stage
                index="05"
                title="Cross-encoder"
                subtitle={text(baseline.perspective, "未執行")}
                duration={number(timings.reranker)}
                active={Boolean(selected.cross_encoder)}
              />
              <Stage
                index="06"
                title="Final judge"
                subtitle={text(
                  selected.llm_reranker?.perspective,
                  "未啟用",
                )}
                duration={number(timings.final_judge)}
                active={number(timings.final_judge) > 0}
              />
              <Stage
                index="07"
                title="Delivery"
                subtitle={resultDelivery}
                duration={number(timings.image)}
              />
            </div>
          </section>

          <div className="analysis-grid">
            <section className="panel planner-panel">
              <div className="section-heading compact">
                <div>
                  <span>PLANNER TRACE</span>
                  <h3>規劃與校正</h3>
                </div>
                {typeof planner.confidence === "number" && (
                  <strong>{Math.round(planner.confidence * 100)}%</strong>
                )}
              </div>
              {planner.action ? (
                <>
                  <div className="comparison">
                    <div>
                      <small>RAW GOAL</small>
                      <p>{text(rawPlanner.reaction_goal)}</p>
                    </div>
                    <span>→</span>
                    <div>
                      <small>FINAL GOAL</small>
                      <p>{text(planner.reaction_goal)}</p>
                    </div>
                  </div>
                  <div className="planner-facts">
                    <div>
                      <small>說話視角</small>
                      <strong>{text(planner.speaker_perspective)}</strong>
                    </div>
                    <div>
                      <small>梗圖功能</small>
                      <strong>{text(planner.meme_role)}</strong>
                    </div>
                  </div>
                  <div className="chip-row">
                    {list(planner.search_terms).map((term) => (
                      <Tag key={String(term)}>{String(term)}</Tag>
                    ))}
                  </div>
                  <div className="reason-box">
                    <small>模型理由</small>
                    <p>{text(planner.reason)}</p>
                  </div>
                  <div className="adjustments">
                    <small>程式校正</small>
                    {adjustments.length ? (
                      adjustments.map((adjustment) => (
                        <Tag key={String(adjustment)} tone="orange">
                          {String(adjustment)}
                        </Tag>
                      ))
                    ) : (
                      <span>沒有套用校正</span>
                    )}
                  </div>
                </>
              ) : (
                <div className="empty compact-empty">
                  訊息在前置規則階段就決定沉默，沒有呼叫 Planner。
                </div>
              )}
            </section>

            <section className="panel retrieval-panel">
              <div className="section-heading compact">
                <div>
                  <span>RETRIEVAL TRACE</span>
                  <h3>召回依據</h3>
                </div>
                <Tag>{text(retrieval.mode, "未執行")}</Tag>
              </div>
              {list(retrieval.queries).length ? (
                <ol className="query-list">
                  {list(retrieval.queries).map((item, index) => (
                    <li key={`${String(item)}-${index}`}>
                      <span>Q{index + 1}</span>
                      <p>{String(item)}</p>
                    </li>
                  ))}
                </ol>
              ) : (
                <div className="empty compact-empty">沒有建立召回查詢。</div>
              )}
              <div className="lexical">
                <small>LEXICAL TERMS</small>
                <div className="chip-row">
                  {list(retrieval.lexical_terms).map((term) => (
                    <Tag key={String(term)}>{String(term)}</Tag>
                  ))}
                  {!list(retrieval.lexical_terms).length && <span>—</span>}
                </div>
              </div>
            </section>
          </div>

          <section className="candidate-section">
            <div className="section-heading">
              <div>
                <span>CANDIDATE EVIDENCE</span>
                <h3>候選圖片比較</h3>
              </div>
              <strong>{candidates.length} candidates</strong>
            </div>
            {candidates.length ? (
              <div className="candidate-table">
                <div className="candidate-row candidate-header">
                  <span>排名</span>
                  <span>字幕</span>
                  <span>Semantic</span>
                  <span>Context</span>
                  <span>Reaction</span>
                  <span>結果</span>
                </div>
                {candidates.map((candidate, index) => {
                  const selectedCandidate =
                    String(candidate.segment_id) === String(selectedSegment);
                  const contextScore = number(
                    candidate.contextual_reranker_score,
                    -1,
                  );
                  const reactionScore = number(
                    candidate.reaction_reranker_score,
                    -1,
                  );
                  const score = candidateScore(candidate);
                  return (
                    <div
                      className={`candidate-row ${
                        selectedCandidate ? "candidate-selected" : ""
                      }`}
                      key={`${String(candidate.segment_id)}-${index}`}
                    >
                      <span className="rank">
                        {String(candidate.contextual_reranker_rank ?? index + 1).padStart(
                          2,
                          "0",
                        )}
                      </span>
                      <div className="caption">
                        <strong>{text(candidate.text)}</strong>
                        <small>
                          segment {text(candidate.segment_id)} · pool{" "}
                          {text(candidate.retrieval_pool_index)}
                        </small>
                      </div>
                      <Score value={number(candidate.semantic_score, -1)} />
                      <Score value={contextScore} />
                      <Score value={reactionScore} />
                      <span>
                        {selectedCandidate ? (
                          <Tag tone="green">SELECTED</Tag>
                        ) : score === maxScore ? (
                          <Tag tone="orange">TOP SCORE</Tag>
                        ) : (
                          <span className="muted">—</span>
                        )}
                      </span>
                    </div>
                  );
                })}
              </div>
            ) : (
              <div className="empty">
                這筆訊息在候選召回前已決定保持沉默，因此沒有圖片候選。
              </div>
            )}
          </section>

          <footer className="trace-footer">
            <span>
              <i />
              {datasetName.startsWith("CT108")
                ? `直接讀取 CT108 · ${lastUpdated || "同步中"}`
                : "內容來自瀏覽器本機匯入"}
            </span>
            <code>{selected.selection_id}</code>
          </footer>
        </article>
      </section>
    </main>
  );
}

function Score({ value }: { value: number }) {
  if (value < 0) return <span className="muted">—</span>;
  const percent = Math.max(3, Math.min(100, value * 100));
  return (
    <div className="score">
      <strong>{value.toFixed(2)}</strong>
      <span>
        <i style={{ width: `${percent}%` }} />
      </span>
    </div>
  );
}
