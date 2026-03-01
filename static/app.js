const runButton = document.getElementById("run-btn");
const resultsBody = document.querySelector("#results-table tbody");
const terminalLog = document.getElementById("terminal-log");
const streamState = document.getElementById("stream-state");
const streamMeta = document.getElementById("stream-meta");
const progressFill = document.getElementById("progress-fill");
const trendModeEl = document.getElementById("trend-mode");
const logRateEl = document.getElementById("log-rate");
const chartModalEl = document.getElementById("chart-modal");
const chartModalCanvas = document.getElementById("chart-modal-canvas");
const chartModalTitle = document.getElementById("chart-modal-title");
const chartModalClose = document.getElementById("chart-modal-close");
const chartModalCloseBtn = document.getElementById("chart-modal-close-btn");
const narrationLog = document.getElementById("narration-log");
const narrationRate = document.getElementById("narration-rate");
const narrationModeEl = document.getElementById("narration-mode");
const playbackAlgoEl = document.getElementById("playback-algo");
const playbackSliderEl = document.getElementById("playback-slider");
const playbackMetaEl = document.getElementById("playback-meta");
const fairnessAuditBody = document.querySelector("#fairness-audit-table tbody");
const fairnessAlertEl = document.getElementById("fairness-alert");
const counterfactualBody = document.querySelector(
  "#counterfactual-table tbody",
);
const counterfactualSummaryEl = document.getElementById(
  "counterfactual-summary",
);
const scenarioRunBtn = document.getElementById("scenario-run-btn");
const ablationRunBtn = document.getElementById("ablation-run-btn");
const counterfactualBtn = document.getElementById("counterfactual-btn");
const reportBtn = document.getElementById("report-btn");
const scenarioSummaryEl = document.getElementById("scenario-summary");
const ablationSummaryEl = document.getElementById("ablation-summary");
const scenarioRunStatusEl = document.getElementById("scenario-run-status");
const scenarioElapsedEl = document.getElementById("scenario-elapsed");
const ablationRunStatusEl = document.getElementById("ablation-run-status");
const ablationElapsedEl = document.getElementById("ablation-elapsed");
const patientSuggestBtn = document.getElementById("patient-suggest-btn");
const patientSummaryEl = document.getElementById("patient-summary");
const sectionNavButtons = [
  ...document.querySelectorAll(".section-nav-btn[data-section]"),
];
const sectionPanels = [
  ...document.querySelectorAll(".section-panel[data-section]"),
];

const palette = {
  clinical: "#66d9ef",
  cost_sensitive: "#a6e22e",
  multi_objective: "#ffd866",
  primal_dual: "#ff6188",
  fairness_constrained: "#00ff88",
};

let eventSource = null;
let modalChart = null;
let runStartedAt = null;
let heartbeatTimer = null;
let logCount = 0;
let autotuneProgressCounter = 0;
let streamCompleted = false;
let expectedAlgorithms = 0;
let scheduledRealtimeKpiRefresh = false;
let logFlushTimer = null;
let logBuffer = [];
let logLineCount = 0;
let recentLogCount = 0;
let lastLogRateTick = Date.now();
let narrationCount = 0;
const MAX_LOG_LINES = 12000;
const LOG_FLUSH_MS = 20;
const chartRegistry = {};
const liveSeries = {};
const resultRegistry = {};
const progressTracker = {};
let scheduledLineRefresh = false;

const telemetry = {
  streamStartMs: 0,
  lastProgressMs: 0,
  progressEvents: 0,
};

const simulationContext = {
  lastAblation: null,
  lastScenarioLab: null,
};

const runTimers = {
  scenario: { intervalId: null, startedAt: 0 },
  ablation: { intervalId: null, startedAt: 0 },
};

function setRunTimer(mode, running) {
  const timer = runTimers[mode];
  if (!timer) {
    return;
  }
  const statusEl =
    mode === "scenario" ? scenarioRunStatusEl : ablationRunStatusEl;
  const elapsedEl = mode === "scenario" ? scenarioElapsedEl : ablationElapsedEl;

  if (running) {
    timer.startedAt = Date.now();
    if (statusEl) {
      statusEl.classList.add("running");
    }
    if (elapsedEl) {
      elapsedEl.textContent = "Elapsed: 0.0s";
    }
    if (timer.intervalId) {
      clearInterval(timer.intervalId);
    }
    timer.intervalId = setInterval(() => {
      if (!elapsedEl) {
        return;
      }
      const elapsedSeconds = (Date.now() - timer.startedAt) / 1000;
      elapsedEl.textContent = `Elapsed: ${elapsedSeconds.toFixed(1)}s`;
    }, 100);
    return;
  }

  if (timer.intervalId) {
    clearInterval(timer.intervalId);
    timer.intervalId = null;
  }
  const elapsedSeconds = timer.startedAt
    ? (Date.now() - timer.startedAt) / 1000
    : 0;
  if (elapsedEl) {
    elapsedEl.textContent = `Elapsed: ${elapsedSeconds.toFixed(1)}s`;
  }
  if (statusEl) {
    statusEl.classList.remove("running");
  }
}

function setActiveSection(sectionId) {
  if (!sectionId) {
    return;
  }
  sectionPanels.forEach((panel) => {
    panel.classList.toggle("active", panel.dataset.section === sectionId);
  });
  sectionNavButtons.forEach((button) => {
    button.classList.toggle("active", button.dataset.section === sectionId);
  });
}

function initSectionNavigation() {
  sectionNavButtons.forEach((button) => {
    button.addEventListener("click", () => {
      setActiveSection(button.dataset.section);
    });
  });
  const defaultSection =
    sectionNavButtons.find((button) => button.classList.contains("active"))
      ?.dataset.section || "live";
  setActiveSection(defaultSection);
}

function renderCounterfactualResult(data, summaryEl = counterfactualSummaryEl) {
  if (!data) {
    return;
  }
  if (summaryEl) {
    summaryEl.textContent = `${data.why} Baseline actions: ${JSON.stringify(data.baseline_actions)}`;
  }
  if (counterfactualBody) {
    counterfactualBody.innerHTML = "";
    (data.action_table || []).forEach((row) => {
      const tr = document.createElement("tr");
      tr.innerHTML = `<td>${row.action}</td><td>${format(row.proxy_reward)}</td><td>${format(row.expected_clinical_delta)}</td><td>${format(row.projected_rfb)}</td>`;
      counterfactualBody.appendChild(tr);
    });
  }
}

function meanValue(values) {
  if (!values.length) return 0;
  return values.reduce((acc, value) => acc + value, 0) / values.length;
}

function stdValue(values) {
  if (!values.length) return 0;
  const mu = meanValue(values);
  const variance = meanValue(values.map((value) => (value - mu) ** 2));
  return Math.sqrt(Math.max(variance, 0));
}

function slopeValue(values) {
  if (values.length < 2) return 0;
  const n = values.length;
  const xMean = (n - 1) / 2;
  const yMean = meanValue(values);
  let num = 0;
  let den = 0;
  for (let index = 0; index < n; index += 1) {
    const dx = index - xMean;
    num += dx * (values[index] - yMean);
    den += dx * dx;
  }
  return den > 0 ? num / den : 0;
}

function buildPayload() {
  const selectedAlgorithms = [
    ...document.querySelectorAll("#algo-grid input:checked"),
  ].map((item) => item.value);
  return {
    episodes: Number(document.getElementById("episodes").value),
    horizon: Number(document.getElementById("horizon").value),
    cohort_size: Number(document.getElementById("cohort_size").value),
    fairness_lambda: Number(document.getElementById("fairness_lambda").value),
    cost_weight: Number(document.getElementById("cost_weight").value),
    risk_weight: Number(document.getElementById("risk_weight").value),
    seed: Number(document.getElementById("seed").value),
    scenario: document.getElementById("scenario-left")?.value || "baseline",
    drift: {
      cost_inflation: Number(
        document.getElementById("drift-cost-inflation")?.value || 0,
      ),
      adherence_drop: Number(
        document.getElementById("drift-adherence-drop")?.value || 0,
      ),
      efficacy_drop: Number(
        document.getElementById("drift-efficacy-drop")?.value || 0,
      ),
    },
    ablations: {
      fairness_term: Boolean(
        document.getElementById("ablation-fairness-term")?.checked,
      ),
      shielding: Boolean(
        document.getElementById("ablation-shielding")?.checked,
      ),
      replay: Boolean(document.getElementById("ablation-replay")?.checked),
      double_q: Boolean(document.getElementById("ablation-double-q")?.checked),
      episode_shaping: Boolean(
        document.getElementById("ablation-episode-shaping")?.checked,
      ),
    },
    uncertainty_seeds: Number(
      document.getElementById("uncertainty-seeds")?.value || 3,
    ),
    score_weights: {
      reward: Number(document.getElementById("weight-reward")?.value || 1.0),
      clinical: Number(
        document.getElementById("weight-clinical")?.value || 1.2,
      ),
      fairness: Number(
        document.getElementById("weight-fairness")?.value || 1.5,
      ),
      cost: Number(document.getElementById("weight-cost")?.value || 0.8),
    },
    auto_tune: true,
    selected_algorithms: selectedAlgorithms,
  };
}

function format(value, digits = 3) {
  return Number(value).toFixed(digits);
}

function appendLog(line) {
  const now = Date.now();
  const prefix = new Date(now).toLocaleTimeString("en-GB", {
    hour12: false,
    minute: "2-digit",
    second: "2-digit",
  });
  logBuffer.push(`[${prefix}] ${line}`);
  if (!logFlushTimer) {
    logFlushTimer = setTimeout(flushLogs, LOG_FLUSH_MS);
  }
}

function appendNarration(line) {
  if (!narrationLog) {
    return;
  }
  narrationCount += 1;
  if (narrationRate) {
    narrationRate.textContent = `${narrationCount} insights`;
  }
  narrationLog.textContent += `\n${line}`;
  const lines = narrationLog.textContent.split("\n");
  if (lines.length > 400) {
    narrationLog.textContent = lines.slice(-400).join("\n");
  }
  narrationLog.scrollTop = narrationLog.scrollHeight;
}

function flushLogs() {
  if (!logBuffer.length) {
    logFlushTimer = null;
    return;
  }

  const chunk = logBuffer.join("\n");
  const prepend = terminalLog.textContent ? "\n" : "";
  terminalLog.textContent += `${prepend}${chunk}`;

  const added = logBuffer.length;
  logBuffer = [];
  logCount += added;
  logLineCount += added;
  recentLogCount += added;

  const now = Date.now();
  const elapsed = now - lastLogRateTick;
  if (elapsed >= 1000) {
    const perSec = (recentLogCount * 1000) / elapsed;
    logRateEl.textContent = `${logCount} logs | ${format(perSec, 1)}/s`;
    recentLogCount = 0;
    lastLogRateTick = now;
  } else {
    logRateEl.textContent = `${logCount} logs`;
  }

  if (logLineCount > MAX_LOG_LINES) {
    const lines = terminalLog.textContent.split("\n");
    const trimmed = lines.slice(-MAX_LOG_LINES);
    terminalLog.textContent = trimmed.join("\n");
    logLineCount = trimmed.length;
  }

  terminalLog.scrollTop = terminalLog.scrollHeight;
  logFlushTimer = null;
}

function resetRuntime() {
  Object.keys(chartRegistry).forEach((key) => chartRegistry[key]?.destroy());
  Object.keys(chartRegistry).forEach((key) => delete chartRegistry[key]);
  Object.keys(liveSeries).forEach((key) => delete liveSeries[key]);
  Object.keys(resultRegistry).forEach((key) => delete resultRegistry[key]);
  Object.keys(progressTracker).forEach((key) => delete progressTracker[key]);
  scheduledLineRefresh = false;

  resultsBody.innerHTML = "";
  progressFill.style.width = "0%";
  streamState.textContent = "Connecting...";
  streamMeta.textContent = "Initializing stream";
  terminalLog.textContent = "$ boot_simulation()";
  logCount = 1;
  logLineCount = 1;
  logRateEl.textContent = "1 logs";
  runStartedAt = Date.now();
  autotuneProgressCounter = 0;
  streamCompleted = false;
  telemetry.streamStartMs = Date.now();
  telemetry.lastProgressMs = 0;
  telemetry.progressEvents = 0;
  logBuffer = [];
  recentLogCount = 0;
  lastLogRateTick = Date.now();
  narrationCount = 0;
  if (narrationRate) narrationRate.textContent = "0 insights";
  if (narrationLog) narrationLog.textContent = "$ narration_ready...";
  if (fairnessAuditBody) fairnessAuditBody.innerHTML = "";
  if (fairnessAlertEl) fairnessAlertEl.textContent = "No audit data yet.";
  if (counterfactualBody) counterfactualBody.innerHTML = "";
  if (counterfactualSummaryEl)
    counterfactualSummaryEl.textContent = "No explanation yet.";
  if (patientSummaryEl)
    patientSummaryEl.textContent = "No patient recommendation yet.";

  const liveRewardEl = document.getElementById("kpi-live-reward");
  const liveGiniEl = document.getElementById("kpi-live-gini");
  const liveLeaderEl = document.getElementById("kpi-live-leader");
  const throughputEl = document.getElementById("kpi-throughput");
  const activeEl = document.getElementById("kpi-active");
  const etaEl = document.getElementById("kpi-eta");
  const noveltyIndexEl = document.getElementById("kpi-novelty-index");
  const paretoWinsEl = document.getElementById("kpi-pareto-wins");
  const compositeLeadEl = document.getElementById("kpi-composite-lead");
  const fairCostEffEl = document.getElementById("kpi-fair-cost-eff");
  const convVelEl = document.getElementById("kpi-convergence-velocity");
  if (liveRewardEl) liveRewardEl.textContent = "-";
  if (liveGiniEl) liveGiniEl.textContent = "-";
  if (liveLeaderEl) liveLeaderEl.textContent = "-";
  if (throughputEl) throughputEl.textContent = "-";
  if (activeEl) activeEl.textContent = "0";
  if (etaEl) etaEl.textContent = "-";
  if (noveltyIndexEl) noveltyIndexEl.textContent = "-";
  if (paretoWinsEl) paretoWinsEl.textContent = "-";
  if (compositeLeadEl) compositeLeadEl.textContent = "-";
  if (fairCostEffEl) fairCostEffEl.textContent = "-";
  if (convVelEl) convVelEl.textContent = "-";

  if (heartbeatTimer) {
    clearInterval(heartbeatTimer);
    heartbeatTimer = null;
  }

  heartbeatTimer = setInterval(() => {
    if (!eventSource) {
      return;
    }
    const activeAlgos = Object.keys(progressTracker).length;
    const progressPct = progressFill.style.width || "0%";
    const elapsed = Math.max(
      0.001,
      (Date.now() - telemetry.streamStartMs) / 1000,
    );
    const throughput = telemetry.progressEvents / elapsed;
    const eta = document.getElementById("kpi-eta")?.textContent || "-";
    appendLog(
      `$ heartbeat active_algorithms=${activeAlgos} progress=${progressPct} throughput=${format(throughput, 2)}ep/s eta=${eta}`,
    );
    scheduleRealtimeKpiRefresh();
  }, 1000);
}

function scheduleRealtimeKpiRefresh() {
  if (scheduledRealtimeKpiRefresh) {
    return;
  }
  scheduledRealtimeKpiRefresh = true;
  requestAnimationFrame(() => {
    updateRealtimeKpis();
    scheduledRealtimeKpiRefresh = false;
  });
}

function updateRealtimeKpis() {
  const entries = Object.entries(liveSeries);
  const activeCount = Object.keys(progressTracker).length;
  const activeEl = document.getElementById("kpi-active");
  if (activeEl) {
    activeEl.textContent = `${activeCount}/${Math.max(expectedAlgorithms, activeCount)}`;
  }

  const elapsed = Math.max(
    0.001,
    (Date.now() - telemetry.streamStartMs) / 1000,
  );
  const throughput = telemetry.progressEvents / elapsed;
  const throughputEl = document.getElementById("kpi-throughput");
  if (throughputEl) {
    throughputEl.textContent = `${format(throughput, 2)} ep/s`;
  }

  const snapshots = entries
    .map(([algorithm_id, series]) => {
      const idx = series.reward.length - 1;
      if (idx < 0) return null;
      return {
        algorithm_id,
        algorithm: series.algorithm,
        reward: series.reward[idx],
        gini: series.gini[idx],
        adherence: series.adherence[idx],
        clinical: series.clinical[idx],
      };
    })
    .filter(Boolean);

  const liveRewardEl = document.getElementById("kpi-live-reward");
  const liveGiniEl = document.getElementById("kpi-live-gini");
  const liveLeaderEl = document.getElementById("kpi-live-leader");

  if (!snapshots.length) {
    if (liveRewardEl) liveRewardEl.textContent = "-";
    if (liveGiniEl) liveGiniEl.textContent = "-";
    if (liveLeaderEl) liveLeaderEl.textContent = "-";
  } else {
    const meanReward =
      snapshots.reduce((acc, row) => acc + row.reward, 0) / snapshots.length;
    const meanGini =
      snapshots.reduce((acc, row) => acc + row.gini, 0) / snapshots.length;
    const leader = [...snapshots].sort((a, b) => b.reward - a.reward)[0];

    if (liveRewardEl) liveRewardEl.textContent = format(meanReward);
    if (liveGiniEl) liveGiniEl.textContent = format(meanGini);
    if (liveLeaderEl) {
      liveLeaderEl.textContent = `${leader.algorithm} (${format(leader.reward)})`;
    }
  }

  const progressEntries = Object.values(progressTracker);
  const ratio = progressEntries.length
    ? progressEntries.reduce(
        (acc, item) => acc + item.episode / Math.max(1, item.episodes),
        0,
      ) / progressEntries.length
    : 0;
  const etaEl = document.getElementById("kpi-eta");
  if (!etaEl) {
    return;
  }
  if (ratio <= 0.0001 || streamCompleted) {
    etaEl.textContent = streamCompleted ? "0.0s" : "-";
  } else {
    const remaining = (elapsed * (1 - ratio)) / ratio;
    etaEl.textContent = `${format(Math.max(0, remaining), 1)}s`;
  }

  const proposedLive = liveSeries.fairness_constrained?.reward || [];
  const convergenceEl = document.getElementById("kpi-convergence-velocity");
  if (convergenceEl) {
    const tail = proposedLive.slice(-14);
    convergenceEl.textContent =
      tail.length >= 2 ? format(slopeValue(tail), 4) : "-";
  }
}

function makeDataset(algoId, label, data, fill = false) {
  return {
    label,
    data,
    borderColor: palette[algoId] || "#8df8b3",
    backgroundColor: fill
      ? `${palette[algoId] || "#8df8b3"}33`
      : palette[algoId] || "#8df8b3",
    borderWidth: algoId === "fairness_constrained" ? 3 : 2,
    pointRadius: 0,
    tension: 0.2,
    fill,
  };
}

function lineAlgorithmsByMode() {
  const mode = trendModeEl?.value || "focus";
  const all = Object.keys(liveSeries);
  if (mode === "all") {
    return all;
  }

  const proposed = "fairness_constrained";
  const sortedResults = Object.values(resultRegistry).sort(
    (a, b) => b.reward - a.reward,
  );
  const leader = sortedResults[0]?.algorithm_id || all[0];

  const allowed = new Set([proposed, leader]);
  if (sortedResults[1]) {
    allowed.add(sortedResults[1].algorithm_id);
  }
  return all.filter((id) => allowed.has(id));
}

function baseChartOptions() {
  return {
    responsive: true,
    maintainAspectRatio: false,
    animation: false,
    normalized: true,
    plugins: {
      decimation: {
        enabled: true,
        algorithm: "lttb",
        samples: 140,
      },
      legend: {
        labels: {
          color: "#b4ffd0",
          font: { family: "Consolas, JetBrains Mono, monospace", size: 11 },
        },
      },
    },
    scales: {
      x: {
        ticks: { color: "#8ecaa2", maxTicksLimit: 10 },
        grid: { color: "rgba(0, 255, 136, 0.10)" },
      },
      y: {
        ticks: { color: "#8ecaa2" },
        grid: { color: "rgba(0, 255, 136, 0.10)" },
      },
    },
  };
}

function scheduleLineChartRefresh() {
  if (scheduledLineRefresh) {
    return;
  }
  scheduledLineRefresh = true;
  requestAnimationFrame(() => {
    refreshLineCharts();
    scheduledLineRefresh = false;
  });
}

function updateGlobalProgress() {
  const entries = Object.values(progressTracker);
  if (!entries.length) {
    progressFill.style.width = "0%";
    return;
  }
  const totalRatio = entries.reduce((acc, current) => {
    if (!current.episodes || current.episodes <= 0) {
      return acc;
    }
    return acc + current.episode / current.episodes;
  }, 0);
  const globalRatio = totalRatio / entries.length;
  progressFill.style.width = `${Math.max(0, Math.min(100, globalRatio * 100))}%`;
}

function initCharts() {
  const emptyLabels = [];

  chartRegistry.reward = new Chart(document.getElementById("reward-chart"), {
    type: "line",
    data: { labels: emptyLabels, datasets: [] },
    options: baseChartOptions(),
  });

  chartRegistry.gini = new Chart(document.getElementById("gini-chart"), {
    type: "line",
    data: { labels: emptyLabels, datasets: [] },
    options: baseChartOptions(),
  });

  chartRegistry.adherence = new Chart(
    document.getElementById("adherence-chart"),
    {
      type: "line",
      data: { labels: emptyLabels, datasets: [] },
      options: baseChartOptions(),
    },
  );

  chartRegistry.clinical = new Chart(
    document.getElementById("clinical-chart"),
    {
      type: "line",
      data: { labels: emptyLabels, datasets: [] },
      options: baseChartOptions(),
    },
  );

  chartRegistry.rewardBar = new Chart(
    document.getElementById("reward-bar-chart"),
    {
      type: "bar",
      data: { labels: [], datasets: [{ label: "Reward", data: [] }] },
      options: baseChartOptions(),
    },
  );

  chartRegistry.scatter = new Chart(document.getElementById("scatter-chart"), {
    type: "scatter",
    data: { datasets: [] },
    options: {
      ...baseChartOptions(),
      scales: {
        x: {
          title: {
            display: true,
            text: "Avg Treatment Cost",
            color: "#8ecaa2",
          },
          ticks: { color: "#8ecaa2" },
          grid: { color: "rgba(0, 255, 136, 0.10)" },
        },
        y: {
          title: { display: true, text: "Equity Index", color: "#8ecaa2" },
          ticks: { color: "#8ecaa2" },
          grid: { color: "rgba(0, 255, 136, 0.10)" },
        },
      },
    },
  });

  chartRegistry.radar = new Chart(document.getElementById("radar-chart"), {
    type: "radar",
    data: {
      labels: ["Reward", "Clinical", "Adherence", "Equity", "Cost Efficiency"],
      datasets: [],
    },
    options: {
      responsive: true,
      scales: {
        r: {
          angleLines: { color: "rgba(0, 255, 136, 0.12)" },
          grid: { color: "rgba(0, 255, 136, 0.12)" },
          pointLabels: { color: "#9be5b8", font: { size: 11 } },
          ticks: { color: "#7fb995", backdropColor: "transparent" },
        },
      },
      plugins: {
        legend: {
          labels: {
            color: "#b4ffd0",
            font: { family: "Consolas, monospace", size: 10 },
          },
        },
      },
    },
  });

  chartRegistry.rfbBar = new Chart(document.getElementById("rfb-bar-chart"), {
    type: "bar",
    data: {
      labels: [],
      datasets: [
        { label: "Avg RFB", data: [] },
        { label: "Gini RFB", data: [] },
      ],
    },
    options: baseChartOptions(),
  });

  chartRegistry.paretoFrontier = new Chart(
    document.getElementById("pareto-frontier-chart"),
    {
      type: "scatter",
      data: { datasets: [] },
      options: {
        ...baseChartOptions(),
        scales: {
          x: {
            title: {
              display: true,
              text: "Avg Treatment Cost",
              color: "#8ecaa2",
            },
            ticks: { color: "#8ecaa2" },
            grid: { color: "rgba(0, 255, 136, 0.10)" },
          },
          y: {
            title: {
              display: true,
              text: "Clinical Improvement",
              color: "#8ecaa2",
            },
            ticks: { color: "#8ecaa2" },
            grid: { color: "rgba(0, 255, 136, 0.10)" },
          },
        },
      },
    },
  );

  chartRegistry.dominance = new Chart(
    document.getElementById("dominance-chart"),
    {
      type: "bar",
      data: { labels: [], datasets: [{ label: "Dominance Margin", data: [] }] },
      options: {
        ...baseChartOptions(),
        indexAxis: "y",
      },
    },
  );

  chartRegistry.noveltyGain = new Chart(
    document.getElementById("novelty-gain-chart"),
    {
      type: "bar",
      data: {
        labels: [
          "Reward Advantage",
          "Clinical Advantage",
          "Equity Advantage",
          "Cost Efficiency Advantage",
        ],
        datasets: [{ label: "Gain", data: [] }],
      },
      options: baseChartOptions(),
    },
  );

  chartRegistry.uncertaintyBand = new Chart(
    document.getElementById("uncertainty-band-chart"),
    {
      type: "line",
      data: { labels: [], datasets: [] },
      options: baseChartOptions(),
    },
  );

  chartRegistry.ablationImpact = new Chart(
    document.getElementById("ablation-impact-chart"),
    {
      type: "bar",
      data: {
        labels: [],
        datasets: [{ label: "Overall Score Impact", data: [] }],
      },
      options: baseChartOptions(),
    },
  );

  chartRegistry.scenarioCompare = new Chart(
    document.getElementById("scenario-compare-chart"),
    {
      type: "bar",
      data: { labels: [], datasets: [] },
      options: baseChartOptions(),
    },
  );

  chartRegistry.playbackTrace = new Chart(
    document.getElementById("playback-trace-chart"),
    {
      type: "line",
      data: { labels: [], datasets: [] },
      options: baseChartOptions(),
    },
  );
}

function ensureSeries(algoId, label) {
  if (!liveSeries[algoId]) {
    liveSeries[algoId] = {
      algorithm: label,
      reward: [],
      gini: [],
      adherence: [],
      clinical: [],
    };
  }
}

function refreshLineCharts() {
  const labels = [];
  const maxLen = Math.max(
    0,
    ...Object.values(liveSeries).map((item) => item.reward.length),
  );
  for (let i = 0; i < maxLen; i += 1) {
    labels.push(i + 1);
  }

  const allowed = new Set(lineAlgorithmsByMode());
  const datasetsByKey = (key, fill = false) =>
    Object.entries(liveSeries)
      .filter(([algoId]) => allowed.has(algoId))
      .map(([algoId, series]) =>
        makeDataset(algoId, series.algorithm, series[key], fill),
      );

  chartRegistry.reward.data.labels = labels;
  chartRegistry.reward.data.datasets = datasetsByKey("reward");
  chartRegistry.gini.data.labels = labels;
  chartRegistry.gini.data.datasets = datasetsByKey("gini");
  chartRegistry.adherence.data.labels = labels;
  chartRegistry.adherence.data.datasets = datasetsByKey("adherence");
  chartRegistry.clinical.data.labels = labels;
  chartRegistry.clinical.data.datasets = datasetsByKey("clinical");

  chartRegistry.reward.update("none");
  chartRegistry.gini.update("none");
  chartRegistry.adherence.update("none");
  chartRegistry.clinical.update("none");
}

function updateExtendedKpis(results) {
  const proposed = results.find(
    (item) => item.algorithm_id === "fairness_constrained",
  );
  const top = results[0];

  const proposedRank = proposed
    ? results.findIndex(
        (item) => item.algorithm_id === "fairness_constrained",
      ) + 1
    : "-";
  const rewardGap = proposed && top ? top.reward - proposed.reward : null;
  const runtimeSeconds = runStartedAt ? (Date.now() - runStartedAt) / 1000 : 0;

  const proposedRankEl = document.getElementById("kpi-proposed-rank");
  const proposedRewardEl = document.getElementById("kpi-proposed-reward");
  const rewardGapEl = document.getElementById("kpi-reward-gap");
  const runtimeEl = document.getElementById("kpi-runtime");

  proposedRankEl.textContent = proposed
    ? `${proposedRank}/${results.length}`
    : "-";
  proposedRewardEl.textContent = proposed ? format(proposed.reward) : "-";
  rewardGapEl.textContent = rewardGap !== null ? format(rewardGap) : "-";
  runtimeEl.textContent = `${runtimeSeconds.toFixed(1)}s`;
}

function renderKpis(results) {
  if (!results.length) return;
  const top = results[0];
  const bestClinical = [...results].sort(
    (a, b) => b.clinical_improvement - a.clinical_improvement,
  )[0];
  const bestEquity = [...results].sort(
    (a, b) => b.equity_index - a.equity_index,
  )[0];
  const lowestGini = [...results].sort((a, b) => a.gini_rfb - b.gini_rfb)[0];

  document.getElementById("kpi-top").textContent = top.algorithm;
  document.getElementById("kpi-clinical").textContent =
    `${bestClinical.algorithm} (${format(bestClinical.clinical_improvement)})`;
  document.getElementById("kpi-equity").textContent =
    `${bestEquity.algorithm} (${format(bestEquity.equity_index)})`;
  document.getElementById("kpi-gini").textContent =
    `${lowestGini.algorithm} (${format(lowestGini.gini_rfb)})`;
}

function renderTable(results) {
  resultsBody.innerHTML = "";
  if (!results.length) return;
  const winner = results[0].algorithm_id;

  results.forEach((row) => {
    const tr = document.createElement("tr");
    if (row.algorithm_id === "fairness_constrained")
      tr.classList.add("proposed");
    tr.innerHTML = `
      <td class="${row.algorithm_id === winner ? "winner" : ""}">${row.algorithm}</td>
      <td>${format(row.reward)}</td>
      <td>${format(row.clinical_improvement)}</td>
      <td>${format(row.adherence)}</td>
      <td>${format(row.avg_treatment_cost, 2)}</td>
      <td>${format(row.avg_rfb)}</td>
      <td>${format(row.gini_rfb)}</td>
      <td>${format(row.equity_index)}</td>
    `;
    resultsBody.appendChild(tr);
  });
}

function compositeWeights() {
  return {
    reward: Number(document.getElementById("weight-reward")?.value || 1),
    clinical: Number(document.getElementById("weight-clinical")?.value || 1),
    fairness: Number(document.getElementById("weight-fairness")?.value || 1),
    cost: Number(document.getElementById("weight-cost")?.value || 1),
  };
}

function sortByComposite(results) {
  const weights = compositeWeights();
  return [...results]
    .map((item) => {
      const composite =
        weights.reward * item.reward +
        weights.clinical * item.clinical_improvement +
        weights.fairness * item.equity_index -
        weights.cost * item.avg_treatment_cost;
      return { ...item, composite_score: composite };
    })
    .sort((a, b) => b.composite_score - a.composite_score);
}

function updateFairnessAuditPanel(results) {
  if (!fairnessAuditBody || !fairnessAlertEl) {
    return;
  }
  const proposed = results.find(
    (item) => item.algorithm_id === "fairness_constrained",
  );
  fairnessAuditBody.innerHTML = "";

  const audit = proposed?.fairness_audit;
  if (!audit || !audit.subgroup_mean_rfb) {
    fairnessAlertEl.textContent = "No fairness audit data yet.";
    return;
  }

  const groups = Object.entries(audit.subgroup_mean_rfb);
  groups.forEach(([group, value]) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${group}</td><td>${format(value, 4)}</td>`;
    fairnessAuditBody.appendChild(tr);
  });

  const gap = Number(audit.subgroup_parity_gap || 0);
  const worst = audit.worst_subgroup || "-";
  fairnessAlertEl.textContent =
    gap > 0.12
      ? `ALERT: Parity gap ${format(gap, 3)} detected. Worst subgroup: ${worst}`
      : `Parity gap ${format(gap, 3)} (stable). Worst subgroup: ${worst}`;
}

function updateUncertaintyChart(results) {
  if (!chartRegistry.uncertaintyBand) {
    return;
  }
  const labels = results.map((item) => item.algorithm);
  const means = results.map(
    (item) => item.uncertainty?.reward?.mean ?? item.reward,
  );
  const lowers = results.map(
    (item) => item.uncertainty?.reward?.lower ?? item.reward,
  );
  const uppers = results.map(
    (item) => item.uncertainty?.reward?.upper ?? item.reward,
  );

  chartRegistry.uncertaintyBand.data.labels = labels;
  chartRegistry.uncertaintyBand.data.datasets = [
    {
      label: "Reward Lower 95%",
      data: lowers,
      borderColor: "rgba(0, 215, 255, 0.4)",
      pointRadius: 0,
      tension: 0.2,
      fill: false,
    },
    {
      label: "Reward Mean",
      data: means,
      borderColor: "#00ff88",
      backgroundColor: "rgba(0,255,136,0.2)",
      pointRadius: 2,
      tension: 0.2,
      fill: false,
    },
    {
      label: "Reward Upper 95%",
      data: uppers,
      borderColor: "rgba(255, 216, 102, 0.4)",
      pointRadius: 0,
      tension: 0.2,
      fill: "-1",
      backgroundColor: "rgba(0, 255, 136, 0.12)",
    },
  ];
  chartRegistry.uncertaintyBand.update();
}

function maybeNarrateProgress(data) {
  if (!narrationModeEl?.checked) {
    return;
  }
  if (data.episode % 20 !== 0 && data.episode !== data.episodes) {
    return;
  }
  appendNarration(
    `$ insight ${data.algorithm} reached ep ${data.episode}/${data.episodes}, reward=${format(data.reward)}, gini=${format(data.gini)}, epsilon=${format(data.epsilon ?? 0, 4)}`,
  );
}

function maybeNarrateCompletion(data) {
  if (!narrationModeEl?.checked) {
    return;
  }
  const proposed = resultRegistry.fairness_constrained;
  if (!proposed) {
    return;
  }
  const baseline = Object.values(resultRegistry)
    .filter((item) => item.algorithm_id !== "fairness_constrained")
    .sort((a, b) => b.reward - a.reward)[0];
  if (!baseline) {
    return;
  }

  const rewardDeltaPct =
    (100 * (proposed.reward - baseline.reward)) /
    Math.max(Math.abs(baseline.reward), 1e-6);
  appendNarration(
    `$ narration Proposed crossed baseline ${baseline.algorithm} by ${format(rewardDeltaPct, 2)}% reward and gini=${format(proposed.gini_rfb, 3)} vs ${format(baseline.gini_rfb, 3)}.`,
  );
}

function updatePlaybackTimeline() {
  const algoId = playbackAlgoEl?.value || "fairness_constrained";
  const series = liveSeries[algoId];
  if (!series) {
    if (playbackMetaEl)
      playbackMetaEl.textContent = "Episode snapshot unavailable";
    return;
  }
  const maxEpisode = Math.max(1, series.reward.length);
  if (playbackSliderEl) {
    playbackSliderEl.max = String(maxEpisode);
    if (Number(playbackSliderEl.value) > maxEpisode) {
      playbackSliderEl.value = String(maxEpisode);
    }
  }
  const episode = Math.max(1, Number(playbackSliderEl?.value || maxEpisode));
  const idx = episode - 1;
  const reward =
    series.reward[idx] ?? series.reward[series.reward.length - 1] ?? 0;
  const gini = series.gini[idx] ?? series.gini[series.gini.length - 1] ?? 0;
  const adherence =
    series.adherence[idx] ?? series.adherence[series.adherence.length - 1] ?? 0;
  const clinical =
    series.clinical[idx] ?? series.clinical[series.clinical.length - 1] ?? 0;
  if (playbackMetaEl) {
    playbackMetaEl.textContent = `Episode ${episode}: reward=${format(reward)} gini=${format(gini)} adh=${format(adherence)} clin=${format(clinical)}`;
  }

  if (chartRegistry.playbackTrace) {
    const labels = Array.from({ length: maxEpisode }, (_, i) => i + 1);
    chartRegistry.playbackTrace.data.labels = labels;
    chartRegistry.playbackTrace.data.datasets = [
      makeDataset(algoId, `${series.algorithm} Reward`, series.reward, false),
      makeDataset(algoId, `${series.algorithm} Gini`, series.gini, false),
    ];
    chartRegistry.playbackTrace.update("none");
  }
}

async function runAblationStudio() {
  const payload = buildPayload();
  const res = await fetch("/api/ablation", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    throw new Error(`ablation_request_failed_${res.status}`);
  }
  const data = await res.json();
  if (!data || !Array.isArray(data.experiments)) {
    throw new Error("ablation_invalid_response");
  }
  simulationContext.lastAblation = data;
  const labels = data.experiments.map((item) => item.component);
  const impacts = data.experiments.map((item) => item.impact.overall_score);
  if (chartRegistry.ablationImpact) {
    chartRegistry.ablationImpact.data.labels = labels;
    chartRegistry.ablationImpact.data.datasets = [
      {
        label: "Overall Score Impact",
        data: impacts,
        backgroundColor: impacts.map((value) =>
          value >= 0 ? "rgba(0,255,136,0.42)" : "rgba(255,56,96,0.42)",
        ),
        borderColor: impacts.map((value) =>
          value >= 0 ? "#00ff88" : "#ff3860",
        ),
        borderWidth: 1,
      },
    ];
    chartRegistry.ablationImpact.update();
  }
  if (ablationSummaryEl) {
    const strongest = data.experiments
      .map((item) => ({
        component: item.component,
        impact: Number(item.impact?.overall_score || 0),
      }))
      .sort((a, b) => Math.abs(b.impact) - Math.abs(a.impact))[0];
    ablationSummaryEl.textContent = strongest
      ? `Strongest impact: ${strongest.component} (${format(strongest.impact)})`
      : "Ablation complete.";
  }
  appendLog(`$ ablation_complete experiments=${data.experiments.length}`);
}

async function runScenarioLab() {
  const base = buildPayload();
  const payload = {
    ...base,
    scenario_left:
      document.getElementById("scenario-left")?.value || "baseline",
    scenario_right:
      document.getElementById("scenario-right")?.value || "low_income_skew",
  };
  const res = await fetch("/api/scenario_lab", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    throw new Error(`scenario_lab_request_failed_${res.status}`);
  }
  const data = await res.json();
  if (!data?.left || !data?.right) {
    throw new Error("scenario_lab_invalid_response");
  }
  simulationContext.lastScenarioLab = data;

  const delta = data.proposed_delta || {};
  if (chartRegistry.scenarioCompare) {
    chartRegistry.scenarioCompare.data.labels = [
      "Reward Δ",
      "Equity Δ",
      "Clinical Δ",
      "Cost Δ",
      "RFB Δ",
    ];
    chartRegistry.scenarioCompare.data.datasets = [
      {
        label: `${data.left.scenario} → ${data.right.scenario}`,
        data: [
          delta.reward || 0,
          delta.equity_index || 0,
          delta.clinical_improvement || 0,
          delta.avg_treatment_cost || 0,
          delta.avg_rfb || 0,
        ],
        backgroundColor: "rgba(0,215,255,0.35)",
        borderColor: "#00d7ff",
        borderWidth: 1,
      },
    ];
    chartRegistry.scenarioCompare.update();
  }
  if (scenarioSummaryEl) {
    scenarioSummaryEl.textContent = `Compared ${data.left.scenario} vs ${data.right.scenario} | Proposed reward Δ=${format(delta.reward || 0)} equity Δ=${format(delta.equity_index || 0)}`;
  }
  appendLog(
    `$ scenario_lab_complete ${data.left.scenario} vs ${data.right.scenario}`,
  );
}

async function runCounterfactualExplainer() {
  const payload = {
    scenario: document.getElementById("scenario-left")?.value || "baseline",
    drift: {
      cost_inflation: Number(
        document.getElementById("drift-cost-inflation")?.value || 0,
      ),
      adherence_drop: Number(
        document.getElementById("drift-adherence-drop")?.value || 0,
      ),
      efficacy_drop: Number(
        document.getElementById("drift-efficacy-drop")?.value || 0,
      ),
    },
    state: {
      severity: Number(document.getElementById("cf-severity")?.value || 0.62),
      cumulative_cost: Number(
        document.getElementById("cf-cum-cost")?.value || 200,
      ),
      last_adherence: Number(
        document.getElementById("cf-last-adh")?.value || 1,
      ),
    },
    profile: {
      economic_capacity: Number(
        document.getElementById("cf-capacity")?.value || 1800,
      ),
      insurance_support: Number(
        document.getElementById("cf-insurance")?.value || 0.28,
      ),
      vulnerability: Number(
        document.getElementById("cf-vulnerability")?.value || 0.52,
      ),
    },
  };
  const res = await fetch("/api/counterfactual", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    throw new Error(`counterfactual_request_failed_${res.status}`);
  }
  const data = await res.json();
  renderCounterfactualResult(data);
  appendLog(`$ counterfactual_done recommended=${data.recommended_action}`);
}

async function runPatientSuggestion() {
  const severity = Number(
    document.getElementById("patient-severity")?.value || 0.55,
  );
  const economicCapacity = Number(
    document.getElementById("patient-capacity")?.value || 1800,
  );
  const adherence = Number(
    document.getElementById("patient-adherence")?.value || 0.8,
  );

  const payload = {
    scenario: document.getElementById("scenario-left")?.value || "baseline",
    drift: {
      cost_inflation: Number(
        document.getElementById("drift-cost-inflation")?.value || 0,
      ),
      adherence_drop: Number(
        document.getElementById("drift-adherence-drop")?.value || 0,
      ),
      efficacy_drop: Number(
        document.getElementById("drift-efficacy-drop")?.value || 0,
      ),
    },
    state: {
      severity,
      cumulative_cost: Math.max(0, (1 - adherence) * 600),
      last_adherence: adherence,
    },
    profile: {
      economic_capacity: economicCapacity,
      insurance_support: 0.28,
      vulnerability: Math.min(0.95, Math.max(0.1, severity * 0.9)),
    },
  };

  const res = await fetch("/api/counterfactual", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    throw new Error(`patient_suggestion_failed_${res.status}`);
  }
  const data = await res.json();

  if (patientSummaryEl) {
    patientSummaryEl.textContent = `Recommended action: ${data.recommended_action}. ${data.why}`;
  }
  renderCounterfactualResult(data, counterfactualSummaryEl);
  appendLog(`$ patient_suggestion recommended=${data.recommended_action}`);
}

function exportDemoReportPdf() {
  const jsPdf = window.jspdf?.jsPDF;
  if (!jsPdf) {
    appendLog("$ report_error jspdf not loaded");
    return;
  }
  const doc = new jsPdf({ orientation: "p", unit: "pt", format: "a4" });
  const now = new Date().toLocaleString();
  doc.setFontSize(14);
  doc.text("Ethical RL Demo Report", 40, 40);
  doc.setFontSize(10);
  doc.text(`Generated: ${now}`, 40, 58);
  doc.text(
    `Top Algorithm: ${document.getElementById("kpi-top")?.textContent || "-"}`,
    40,
    78,
  );
  doc.text(
    `Novelty Index: ${document.getElementById("kpi-novelty-index")?.textContent || "-"}`,
    40,
    94,
  );
  doc.text(
    `Pareto Wins: ${document.getElementById("kpi-pareto-wins")?.textContent || "-"}`,
    40,
    110,
  );
  doc.text(
    `Composite Lead: ${document.getElementById("kpi-composite-lead")?.textContent || "-"}`,
    40,
    126,
  );
  doc.text(`Fairness Alert: ${fairnessAlertEl?.textContent || "-"}`, 40, 142, {
    maxWidth: 520,
  });

  const chart = chartRegistry.rewardBar;
  if (chart?.toBase64Image) {
    try {
      doc.addImage(chart.toBase64Image(), "PNG", 40, 170, 500, 220);
    } catch {
      // ignore image capture failures
    }
  }
  doc.save("ethical-rl-demo-report.pdf");
  appendLog("$ report_exported ethical-rl-demo-report.pdf");
}

function updateSummaryCharts(results) {
  const labels = results.map((item) => item.algorithm);
  const colors = results.map((item) => palette[item.algorithm_id] || "#90ffbc");

  chartRegistry.rewardBar.data.labels = labels;
  chartRegistry.rewardBar.data.datasets = [
    {
      label: "Reward",
      data: results.map((item) => item.reward),
      backgroundColor: colors,
      borderColor: colors,
      borderWidth: 1,
    },
  ];
  chartRegistry.rewardBar.update();

  chartRegistry.scatter.data.datasets = results.map((item) => ({
    label: item.algorithm,
    data: [{ x: item.avg_treatment_cost, y: item.equity_index }],
    backgroundColor: palette[item.algorithm_id] || "#90ffbc",
    borderColor: palette[item.algorithm_id] || "#90ffbc",
    pointRadius: item.algorithm_id === "fairness_constrained" ? 7 : 5,
  }));
  chartRegistry.scatter.update();

  const maxReward = Math.max(1e-6, ...results.map((item) => item.reward));
  const maxClinical = Math.max(
    1e-6,
    ...results.map((item) => item.clinical_improvement),
  );
  const maxCost = Math.max(
    1e-6,
    ...results.map((item) => item.avg_treatment_cost),
  );

  chartRegistry.radar.data.datasets = results.map((item) => {
    const costEfficiency = 1 - item.avg_treatment_cost / maxCost;
    return {
      label: item.algorithm,
      data: [
        item.reward / maxReward,
        item.clinical_improvement / maxClinical,
        item.adherence,
        item.equity_index,
        costEfficiency,
      ],
      borderColor: palette[item.algorithm_id] || "#90ffbc",
      backgroundColor: `${palette[item.algorithm_id] || "#90ffbc"}22`,
      pointRadius: 2,
      borderWidth: item.algorithm_id === "fairness_constrained" ? 3 : 2,
    };
  });
  chartRegistry.radar.update();

  chartRegistry.rfbBar.data.labels = labels;
  chartRegistry.rfbBar.data.datasets = [
    {
      label: "Avg RFB",
      data: results.map((item) => item.avg_rfb),
      backgroundColor: "rgba(0, 255, 136, 0.40)",
      borderColor: "#00ff88",
      borderWidth: 1,
    },
    {
      label: "Gini RFB",
      data: results.map((item) => item.gini_rfb),
      backgroundColor: "rgba(0, 215, 255, 0.35)",
      borderColor: "#00d7ff",
      borderWidth: 1,
    },
  ];
  chartRegistry.rfbBar.update();

  chartRegistry.paretoFrontier.data.datasets = results.map((item) => ({
    label: item.algorithm,
    data: [{ x: item.avg_treatment_cost, y: item.clinical_improvement }],
    backgroundColor: palette[item.algorithm_id] || "#90ffbc",
    borderColor: palette[item.algorithm_id] || "#90ffbc",
    pointRadius: item.algorithm_id === "fairness_constrained" ? 9 : 5,
    pointStyle:
      item.algorithm_id === "fairness_constrained" ? "star" : "circle",
  }));
  chartRegistry.paretoFrontier.update();

  const proposed = results.find(
    (item) => item.algorithm_id === "fairness_constrained",
  );
  const baselines = results.filter(
    (item) => item.algorithm_id !== "fairness_constrained",
  );
  if (proposed && baselines.length) {
    const dominanceMargins = baselines.map((item) => {
      const margin =
        0.42 * (proposed.reward - item.reward) +
        0.36 * (proposed.equity_index - item.equity_index) +
        0.22 * (proposed.clinical_improvement - item.clinical_improvement);
      return { label: item.algorithm, margin };
    });

    chartRegistry.dominance.data.labels = dominanceMargins.map(
      (item) => item.label,
    );
    chartRegistry.dominance.data.datasets = [
      {
        label: "Dominance Margin",
        data: dominanceMargins.map((item) => item.margin),
        backgroundColor: dominanceMargins.map((item) =>
          item.margin >= 0 ? "rgba(0,255,136,0.42)" : "rgba(255,56,96,0.42)",
        ),
        borderColor: dominanceMargins.map((item) =>
          item.margin >= 0 ? "#00ff88" : "#ff3860",
        ),
        borderWidth: 1,
      },
    ];
    chartRegistry.dominance.update();

    const avgBaselineReward = meanValue(baselines.map((item) => item.reward));
    const avgBaselineClinical = meanValue(
      baselines.map((item) => item.clinical_improvement),
    );
    const avgBaselineEquity = meanValue(
      baselines.map((item) => item.equity_index),
    );
    const baselineEfficiency = meanValue(
      baselines.map(
        (item) => item.equity_index / Math.max(item.avg_treatment_cost, 1e-6),
      ),
    );
    const proposedEfficiency =
      proposed.equity_index / Math.max(proposed.avg_treatment_cost, 1e-6);

    chartRegistry.noveltyGain.data.datasets = [
      {
        label: "Gain",
        data: [
          proposed.reward - avgBaselineReward,
          proposed.clinical_improvement - avgBaselineClinical,
          proposed.equity_index - avgBaselineEquity,
          proposedEfficiency - baselineEfficiency,
        ],
        backgroundColor: [
          "rgba(255,216,102,0.45)",
          "rgba(102,217,239,0.45)",
          "rgba(0,255,136,0.45)",
          "rgba(0,215,255,0.45)",
        ],
        borderColor: ["#ffd866", "#66d9ef", "#00ff88", "#00d7ff"],
        borderWidth: 1,
      },
    ];
    chartRegistry.noveltyGain.update();
  }
}

function updateNoveltyFeatureKpis(results) {
  const proposed = results.find(
    (item) => item.algorithm_id === "fairness_constrained",
  );
  const baselines = results.filter(
    (item) => item.algorithm_id !== "fairness_constrained",
  );

  const noveltyIndexEl = document.getElementById("kpi-novelty-index");
  const paretoWinsEl = document.getElementById("kpi-pareto-wins");
  const compositeLeadEl = document.getElementById("kpi-composite-lead");
  const fairCostEffEl = document.getElementById("kpi-fair-cost-eff");

  if (!proposed || !baselines.length) {
    if (noveltyIndexEl) noveltyIndexEl.textContent = "-";
    if (paretoWinsEl) paretoWinsEl.textContent = "-";
    if (compositeLeadEl) compositeLeadEl.textContent = "-";
    if (fairCostEffEl) fairCostEffEl.textContent = "-";
    return;
  }

  const baselineOverall = meanValue(
    baselines.map((item) => item.overall_score ?? item.reward),
  );
  const proposedOverall = proposed.overall_score ?? proposed.reward;
  const noveltyIndex = proposedOverall - baselineOverall;

  const paretoWins = baselines.filter((item) => {
    const noWorse =
      proposed.reward >= item.reward &&
      proposed.equity_index >= item.equity_index &&
      proposed.clinical_improvement >= item.clinical_improvement;
    const strictlyBetter =
      proposed.reward > item.reward ||
      proposed.equity_index > item.equity_index ||
      proposed.clinical_improvement > item.clinical_improvement;
    return noWorse && strictlyBetter;
  }).length;

  const bestBaselineRank = Math.max(
    ...baselines.map(
      (item) => item.ranking_score ?? item.overall_score ?? item.reward,
    ),
  );
  const proposedRank =
    proposed.ranking_score ?? proposed.overall_score ?? proposed.reward;
  const compositeLead = proposedRank - bestBaselineRank;

  const fairCostEff =
    100 * (proposed.equity_index / Math.max(proposed.avg_treatment_cost, 1e-6));

  if (noveltyIndexEl) noveltyIndexEl.textContent = format(noveltyIndex);
  if (paretoWinsEl)
    paretoWinsEl.textContent = `${paretoWins}/${baselines.length}`;
  if (compositeLeadEl) compositeLeadEl.textContent = format(compositeLead);
  if (fairCostEffEl)
    fairCostEffEl.textContent = `${format(fairCostEff, 2)}x10⁻²`;
}

function hydrateResultsFromRegistry() {
  const sorted = sortByComposite(Object.values(resultRegistry));
  renderTable(sorted);
  renderKpis(sorted);
  updateExtendedKpis(sorted);
  updateNoveltyFeatureKpis(sorted);
  updateFairnessAuditPanel(sorted);
  updateUncertaintyChart(sorted);
  if (sorted.length) {
    updateSummaryCharts(sorted);
  }
  updatePlaybackTimeline();
  scheduleRealtimeKpiRefresh();
}

function cloneChartConfig(baseChart) {
  return {
    type: baseChart.config.type,
    data: JSON.parse(JSON.stringify(baseChart.data)),
    options: JSON.parse(JSON.stringify(baseChart.options)),
  };
}

function openChartModal(chartKey, titleText) {
  const sourceChart = chartRegistry[chartKey];
  if (!sourceChart) {
    return;
  }

  chartModalTitle.textContent = titleText;
  chartModalEl.classList.add("open");
  chartModalEl.setAttribute("aria-hidden", "false");

  if (modalChart) {
    modalChart.destroy();
    modalChart = null;
  }

  const clonedConfig = cloneChartConfig(sourceChart);
  if (clonedConfig.options?.plugins?.decimation) {
    clonedConfig.options.plugins.decimation.enabled = false;
  }
  modalChart = new Chart(chartModalCanvas, clonedConfig);
}

function closeChartModal() {
  chartModalEl.classList.remove("open");
  chartModalEl.setAttribute("aria-hidden", "true");
  if (modalChart) {
    modalChart.destroy();
    modalChart = null;
  }
}

function attachChartExpandHandlers() {
  const mappings = [
    ["reward-chart", "reward", "Learning Curve (Reward)"],
    ["gini-chart", "gini", "Fairness Trend (Gini)"],
    ["adherence-chart", "adherence", "Adherence Trend"],
    ["clinical-chart", "clinical", "Clinical Improvement Trend"],
    ["reward-bar-chart", "rewardBar", "Final Reward Comparison"],
    ["scatter-chart", "scatter", "Cost vs Equity Scatter"],
    ["radar-chart", "radar", "Performance Radar"],
    ["rfb-bar-chart", "rfbBar", "RFB & Gini Bars"],
    [
      "pareto-frontier-chart",
      "paretoFrontier",
      "Pareto Frontier (Cost vs Clinical)",
    ],
    ["dominance-chart", "dominance", "Proposed Dominance Margin"],
    ["novelty-gain-chart", "noveltyGain", "Novelty Gain Decomposition"],
    [
      "uncertainty-band-chart",
      "uncertaintyBand",
      "Uncertainty Bands (Reward CI95)",
    ],
    [
      "ablation-impact-chart",
      "ablationImpact",
      "Ablation Impact (With vs Without)",
    ],
    ["scenario-compare-chart", "scenarioCompare", "Scenario Lab Comparison"],
    ["playback-trace-chart", "playbackTrace", "Playback Fairness/Reward Trace"],
  ];

  mappings.forEach(([canvasId, chartKey, title]) => {
    const canvas = document.getElementById(canvasId);
    if (!canvas) {
      return;
    }
    canvas.onclick = () => openChartModal(chartKey, title);
  });
}

function openSimulationStream() {
  const payloadObj = buildPayload();
  expectedAlgorithms = Array.isArray(payloadObj.selected_algorithms)
    ? payloadObj.selected_algorithms.length
    : 0;
  const payload = encodeURIComponent(JSON.stringify(payloadObj));
  const streamUrl = `/api/simulate_stream?payload=${payload}`;

  eventSource = new EventSource(streamUrl);

  eventSource.addEventListener("start", (event) => {
    const data = JSON.parse(event.data);
    telemetry.streamStartMs = Date.now();
    streamState.textContent = "Streaming";
    streamMeta.textContent = `Episodes: ${data.config.episodes} | Cohort: ${data.config.cohort_size}`;
    appendLog(`$ stream_started episodes=${data.config.episodes}`);
    appendLog(
      `$ scenario=${buildPayload().scenario} drift=${JSON.stringify(buildPayload().drift)}`,
    );
    if (narrationModeEl?.checked) {
      appendNarration(
        `$ narration Run started with scenario=${buildPayload().scenario}, uncertainty_seeds=${buildPayload().uncertainty_seeds}.`,
      );
    }
    if (playbackSliderEl) {
      playbackSliderEl.min = "1";
      playbackSliderEl.max = String(Math.max(1, data.config.episodes || 1));
      playbackSliderEl.value = "1";
    }
    scheduleRealtimeKpiRefresh();
  });

  eventSource.addEventListener("autotune_start", () => {
    streamState.textContent = "Auto-Tuning";
    streamMeta.textContent = "Searching best fairness parameters";
    appendLog("$ autotune_start fairness_constrained");
  });

  eventSource.addEventListener("autotune_progress", (event) => {
    const data = JSON.parse(event.data);
    const tested = Number(data.tested || 0);
    const total = Number(data.total || 0);
    const status = data.status || "running";

    streamState.textContent = "Auto-Tuning";
    streamMeta.textContent =
      total > 0
        ? `Searching best fairness parameters (${tested}/${total})`
        : "Searching best fairness parameters";

    autotuneProgressCounter += 1;
    if (status === "cache_hit") {
      appendLog("$ autotune_cache_hit reusing recent best parameters");
      return;
    }

    if (autotuneProgressCounter % 3 === 0 || tested === total) {
      const params = data.best_params || {};
      appendLog(
        `$ autotune_progress ${tested}/${total} best_lambda=${format(params.fairness_lambda ?? 0, 3)} best_cost_w=${format(params.cost_weight ?? 0, 3)} best_lr=${format(params.learning_rate ?? 0, 3)}`,
      );
    }
  });

  eventSource.addEventListener("autotune_keepalive", () => {
    if (streamState.textContent === "Auto-Tuning") {
      streamMeta.textContent = "Searching best fairness parameters";
    }
  });

  eventSource.addEventListener("autotune_error", (event) => {
    const data = JSON.parse(event.data);
    appendLog(
      `$ autotune_error ${data.message || "failed, using provided parameters"}`,
    );
    streamState.textContent = "Streaming";
    streamMeta.textContent = "Auto-tune failed, fallback parameters applied";
  });

  eventSource.addEventListener("autotune_result", (event) => {
    const data = JSON.parse(event.data);
    const params = data.autotune?.best_params || {};
    appendLog(
      `$ autotune_result lambda=${format(params.fairness_lambda ?? 0, 3)} cost_w=${format(params.cost_weight ?? 0, 3)} lr=${format(params.learning_rate ?? 0, 3)} eps_decay=${format(params.epsilon_decay ?? 0, 4)}`,
    );
    streamState.textContent = "Streaming";
    streamMeta.textContent = "Auto-tune complete, running simulation";
  });

  eventSource.addEventListener("progress", (event) => {
    const data = JSON.parse(event.data);
    telemetry.progressEvents += 1;
    telemetry.lastProgressMs = Date.now();

    ensureSeries(data.algorithm_id, data.algorithm);
    const entry = liveSeries[data.algorithm_id];
    entry.reward.push(data.reward);
    entry.gini.push(data.gini);
    entry.adherence.push(data.adherence);
    entry.clinical.push(data.clinical);

    progressTracker[data.algorithm_id] = {
      episode: data.episode,
      episodes: data.episodes,
    };
    updateGlobalProgress();
    streamMeta.textContent = `${data.algorithm} :: episode ${data.episode}/${data.episodes}`;

    appendLog(
      `$ ${data.algorithm_id} ep=${data.episode}/${data.episodes} reward=${format(data.reward)} gini=${format(data.gini)} adh=${format(data.adherence)} clin=${format(data.clinical)}`,
    );
    maybeNarrateProgress(data);

    const driftActive =
      Number(document.getElementById("drift-cost-inflation")?.value || 0) > 0 ||
      Number(document.getElementById("drift-adherence-drop")?.value || 0) > 0 ||
      Number(document.getElementById("drift-efficacy-drop")?.value || 0) > 0;
    if (driftActive && data.episode % 15 === 0 && data.reward < -0.65) {
      appendNarration(
        `$ robustness_alert ${data.algorithm} under drift shows low reward (${format(data.reward)}). Recovery monitoring active.`,
      );
    }

    if (data.episode % 10 === 0 || data.episode === data.episodes) {
      const elapsed = Math.max(
        0.001,
        (Date.now() - telemetry.streamStartMs) / 1000,
      );
      const throughput = telemetry.progressEvents / elapsed;
      appendLog(
        `$ runtime_snapshot algo=${data.algorithm_id} ep=${data.episode} throughput=${format(throughput, 2)}ep/s total_events=${telemetry.progressEvents}`,
      );
    }

    if (data.episode === data.episodes || data.episode % 2 === 0) {
      scheduleLineChartRefresh();
    }
    updatePlaybackTimeline();
    scheduleRealtimeKpiRefresh();
  });

  eventSource.addEventListener("algorithm_done", (event) => {
    const data = JSON.parse(event.data);
    resultRegistry[data.algorithm_id] = data.result;
    hydrateResultsFromRegistry();
    appendLog(
      `$ algorithm_done ${data.algorithm_id} reward=${format(data.result.reward)} equity=${format(data.result.equity_index)} adherence=${format(data.result.adherence)} clinical=${format(data.result.clinical_improvement)} gini=${format(data.result.gini_rfb)}`,
    );
    maybeNarrateCompletion(data);
  });

  eventSource.addEventListener("algorithm_error", (event) => {
    const data = JSON.parse(event.data);
    appendLog(
      `$ algorithm_error ${data.algorithm_id} ${data.message || "unknown error"}`,
    );
  });

  eventSource.addEventListener("stream_keepalive", (event) => {
    const data = JSON.parse(event.data);
    if (streamState.textContent !== "Completed") {
      streamMeta.textContent = `Running ${data.active_algorithms ?? "?"} algorithm(s)...`;
    }
    scheduleRealtimeKpiRefresh();
  });

  eventSource.addEventListener("stream_error", (event) => {
    const data = JSON.parse(event.data);
    appendLog(`$ stream_error ${data.message || "backend failure"}`);
    streamState.textContent = "Error";
    streamMeta.textContent = "Backend stream error";
    runButton.disabled = false;
    runButton.textContent = "Start";
    streamCompleted = true;
    if (heartbeatTimer) {
      clearInterval(heartbeatTimer);
      heartbeatTimer = null;
    }
    eventSource?.close();
    eventSource = null;
    scheduleRealtimeKpiRefresh();
  });

  eventSource.addEventListener("complete", (event) => {
    const data = JSON.parse(event.data);
    data.results.forEach((item) => {
      resultRegistry[item.algorithm_id] = item;
    });
    hydrateResultsFromRegistry();
    streamState.textContent = "Completed";
    streamMeta.textContent = `${data.results.length} algorithms completed`;
    progressFill.style.width = "100%";
    streamCompleted = true;
    if (Array.isArray(data.errors) && data.errors.length) {
      appendLog(`$ stream_completed_with_errors count=${data.errors.length}`);
    }
    appendLog("$ simulation_complete");
    scheduleLineChartRefresh();
    if (heartbeatTimer) {
      clearInterval(heartbeatTimer);
      heartbeatTimer = null;
    }
    runButton.disabled = false;
    runButton.textContent = "Start";
    eventSource?.close();
    eventSource = null;
    scheduleRealtimeKpiRefresh();
  });

  eventSource.onerror = () => {
    if (eventSource && !streamCompleted) {
      streamState.textContent = "Disconnected";
      appendLog("$ stream_error disconnected");
      runButton.disabled = false;
      runButton.textContent = "Start";
      if (heartbeatTimer) {
        clearInterval(heartbeatTimer);
        heartbeatTimer = null;
      }
      eventSource.close();
      eventSource = null;
      scheduleRealtimeKpiRefresh();
    }
  };
}

function runSimulation() {
  if (eventSource) {
    eventSource.close();
    eventSource = null;
  }

  runButton.disabled = true;
  runButton.textContent = "Streaming...";
  resetRuntime();
  initCharts();
  attachChartExpandHandlers();
  openSimulationStream();
}

runButton.addEventListener("click", runSimulation);
trendModeEl?.addEventListener("change", () => scheduleLineChartRefresh());
playbackAlgoEl?.addEventListener("change", updatePlaybackTimeline);
playbackSliderEl?.addEventListener("input", updatePlaybackTimeline);
scenarioRunBtn?.addEventListener("click", async () => {
  scenarioRunBtn.disabled = true;
  setRunTimer("scenario", true);
  try {
    await runScenarioLab();
  } catch (error) {
    appendLog(`$ scenario_lab_error ${error}`);
    if (scenarioSummaryEl) {
      scenarioSummaryEl.textContent = `Scenario Lab failed: ${error}`;
    }
  } finally {
    setRunTimer("scenario", false);
    scenarioRunBtn.disabled = false;
  }
});
ablationRunBtn?.addEventListener("click", async () => {
  ablationRunBtn.disabled = true;
  setRunTimer("ablation", true);
  try {
    await runAblationStudio();
  } catch (error) {
    appendLog(`$ ablation_error ${error}`);
    if (ablationSummaryEl) {
      ablationSummaryEl.textContent = `Ablation failed: ${error}`;
    }
  } finally {
    setRunTimer("ablation", false);
    ablationRunBtn.disabled = false;
  }
});
counterfactualBtn?.addEventListener("click", async () => {
  counterfactualBtn.disabled = true;
  try {
    await runCounterfactualExplainer();
  } catch (error) {
    appendLog(`$ counterfactual_error ${error}`);
  } finally {
    counterfactualBtn.disabled = false;
  }
});
patientSuggestBtn?.addEventListener("click", async () => {
  patientSuggestBtn.disabled = true;
  try {
    await runPatientSuggestion();
  } catch (error) {
    appendLog(`$ patient_suggestion_error ${error}`);
    if (patientSummaryEl) {
      patientSummaryEl.textContent = `Patient suggestion failed: ${error}`;
    }
  } finally {
    patientSuggestBtn.disabled = false;
  }
});
reportBtn?.addEventListener("click", exportDemoReportPdf);
["weight-reward", "weight-clinical", "weight-fairness", "weight-cost"].forEach(
  (id) =>
    document.getElementById(id)?.addEventListener("input", () => {
      hydrateResultsFromRegistry();
    }),
);
chartModalClose?.addEventListener("click", closeChartModal);
chartModalCloseBtn?.addEventListener("click", closeChartModal);
initSectionNavigation();
initCharts();
attachChartExpandHandlers();
