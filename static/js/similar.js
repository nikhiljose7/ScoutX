// ======================================================================
//  similar.js - Adapted from reference app.js
// ======================================================================

/* eslint-disable no-unused-vars */
/* global Chart */

//
// ELEMENT SELECTORS
//
const searchBox = document.getElementById("searchBox");
const suggestions = document.getElementById("suggestions");
const getSimilarBtn = document.getElementById("getSimilarBtn");
const similarTableHolder = document.getElementById("similarTableHolder");
const kInput = document.getElementById("kInput");
const leagueFilter = document.getElementById("leagueFilter");
const positionFilter = document.getElementById("positionFilter");
const minAgeInput = document.getElementById("minAge");
const maxAgeInput = document.getElementById("maxAge");

const inputPlayerContent = document.getElementById("inputPlayerContent");
const selectedMultipleHolder = document.getElementById("selectedMultipleHolder");
const compareSelectedBtn = document.getElementById("compareSelectedBtn");
const toggleAllFeaturesBtn = document.getElementById("toggleAllFeaturesBtn");
const removeSelectionBtn = document.getElementById("removeSelectionBtn");
const aiReportBox = document.getElementById("aiReportBox");
const toggleAiReportBtn = document.getElementById("toggleAiReportBtn");

let customTooltip = document.getElementById("customTooltip"); // external tooltip div
if (!customTooltip) {
    customTooltip = document.createElement("div");
    customTooltip.id = "customTooltip";
    customTooltip.className = "chartjs-tooltip";
    document.body.appendChild(customTooltip);
}

//
// STATE
//
let selectedPlayerId = null;
let selectedPlayerName = null;
let lastSimilarResults = [];
let lastSimilarInputRadar = null;
let radarChart = null;
let datasetMap = {}; // maps player Rk -> dataset index (chart dataset index)
let featureDescriptions = {};
let pulsateState = { activeId: null, raf: null, startTime: null };
let lastCompareResponse = null;
let showAllFeatures = false;

// For sticky tooltip logic
let pointerInsideCanvas = false;
let pointerInsideTooltip = false;
let hideTooltipTimeout = null;

//
// HELPERS
//
function showError(msg) {
    console.error(msg);
    alert(msg);
}

function debounce(fn, delay = 300) {
    let t;
    return (...args) => {
        clearTimeout(t);
        t = setTimeout(() => fn(...args), delay);
    };
}

function setAlphaRGBA(rgbaStr, newAlpha) {
    try {
        const m = rgbaStr.match(
            /rgba?\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)(?:\s*,\s*([0-9.]+))?\s*\)/i
        );
        if (!m) return rgbaStr;
        const r = m[1],
            g = m[2],
            b = m[3];
        return `rgba(${r}, ${g}, ${b}, ${newAlpha})`;
    } catch (e) {
        return rgbaStr;
    }
}

//
// INIT META: filters + feature descriptions
//
async function initMeta() {
    try {
        const r = await fetch("/api/meta");
        const j = await r.json();
        if (j.ok) {
            (j.leagues || []).forEach((l) => {
                const o = document.createElement("option");
                o.value = l;
                o.innerText = l;
                leagueFilter.appendChild(o);
            });
            (j.positions || []).forEach((p) => {
                const o = document.createElement("option");
                o.value = p;
                o.innerText = p;
                positionFilter.appendChild(o);
            });
        }
    } catch (e) {
        console.warn(e);
    }

    try {
        const r = await fetch("/api/feature_desc");
        const j = await r.json();
        if (j.ok) featureDescriptions = j.descriptions || {};
    } catch (e) {
        console.warn(e);
    }
}
initMeta();

//
// SEARCH
//
searchBox.addEventListener(
    "input",
    debounce(async (e) => {
        const q = e.target.value;
        if (!q || q.length < 2) {
            suggestions.innerHTML = "";
            return;
        }
        try {
            const r = await fetch(`/api/search?q=${encodeURIComponent(q)}&rows=10`);
            const j = await r.json();
            if (j.ok) {
                suggestions.innerHTML = j.results
                    .map(
                        (p) =>
                            `<div class="sugg" data-id="${p.player_id}" data-name="${p.player_name}">${p.player_name}</div>`
                    )
                    .join("");
            }
        } catch (err) {
            console.warn(err);
        }
    }, 200)
);

suggestions.addEventListener("click", async (e) => {
    const t = e.target.closest(".sugg");
    if (!t) return;
    suggestions.innerHTML = "";
    selectedPlayerId = t.dataset.id;
    selectedPlayerName = t.dataset.name;
    searchBox.value = selectedPlayerName;

    try {
        const r = await fetch(
            `/api/player_details?player_id=${encodeURIComponent(selectedPlayerId)}`
        );
        const j = await r.json();
        if (!j.ok) return showError("Couldn't load player.");
        renderInputPlayer(j.player);
    } catch (err) {
        showError(err.message || String(err));
    }
});

//
// INPUT PLAYER PANEL
//
function attachBoxTooltips(container) {
    if (!container) return;
    container.querySelectorAll("[data-key]").forEach((el) => {
        const enter = (e) => {
            const key = el.getAttribute("data-key");
            const desc = featureDescriptions[key] || "";
            if (!customTooltip) return;
            customTooltip.innerHTML = `<div class="tt-title-small">${key}</div><div class="tt-desc-small">${desc}</div>`;
            customTooltip.style.left = e.clientX + 12 + "px";
            customTooltip.style.top = e.clientY + 12 + "px";
            customTooltip.style.display = "block";
        };
        const move = (e) => {
            if (!customTooltip) return;
            customTooltip.style.left = e.clientX + 12 + "px";
            customTooltip.style.top = e.clientY + 12 + "px";
        };
        const leave = () => {
            customTooltip.style.display = "none";
        };
        el.addEventListener("mouseenter", enter);
        el.addEventListener("mousemove", move);
        el.addEventListener("mouseleave", leave);
    });
}

function renderInputPlayer(p) {
    const html = `
    <div class="stat-row"><span class="stat-key">Player</span><span class="stat-value">${p.Player}</span></div>
    <div class="stat-row"><span class="stat-key">Nation</span><span class="stat-value">${p.Nation}</span></div>
    <div class="stat-row"><span class="stat-key">Born</span><span class="stat-value">${p.Born || '-'}</span></div>
    <div class="stat-row"><span class="stat-key">Age</span><span class="stat-value">${p.Age}</span></div>
    <div class="stat-row"><span class="stat-key">Position</span><span class="stat-value">${p.Pos}</span></div>
    <div class="stat-row"><span class="stat-key">Matches</span><span class="stat-value">${p["Playing Time MP"] || p["MP"] || '-'}</span></div>
    <div class="stat-row"><span class="stat-key">Goals</span><span class="stat-value">${p["Performance Gls"] || p["Gls"] || '-'}</span></div>
    <div class="stat-row"><span class="stat-key">Assists</span><span class="stat-value">${p["Performance Ast"] || p["Ast"] || '-'}</span></div>`;
    inputPlayerContent.innerHTML = html;
    attachBoxTooltips(inputPlayerContent);
}

//
// GET SIMILAR
//
getSimilarBtn.addEventListener("click", async () => {
    if (!selectedPlayerId) return showError("Select a player first.");
    const params = new URLSearchParams();
    params.append("player_id", selectedPlayerId);
    params.append("k", kInput.value || 10);
    if (minAgeInput.value) params.append("min_age", minAgeInput.value);
    if (maxAgeInput.value) params.append("max_age", maxAgeInput.value);
    if (leagueFilter.value) params.append("leagues", leagueFilter.value);
    if (positionFilter.value) params.append("positions", positionFilter.value);

    try {
        const r = await fetch(`/api/similar_players?${params.toString()}`);
        const j = await r.json();
        if (!j.ok) return showError("Error fetching similar.");
        lastSimilarResults = j.results;
        lastSimilarInputRadar = j.input_radar;
        renderSimilarTable(j.results);
        renderRadar(j.input_radar, j.results.slice(0, 10), selectedPlayerName);
    } catch (err) {
        showError(err.message || String(err));
    }
});

//
// SIMILAR TABLE
//
function renderSimilarTable(arr) {
    similarTableHolder.innerHTML = `
    <table class="similar-table">
      <thead>
        <tr>
          <th></th><th>#</th><th>Player</th><th>Pos</th><th>Club</th><th>Age</th><th>Sim</th>
        </tr>
      </thead>
      <tbody>
        ${arr
            .map(
                (p, i) => `
          <tr class="srow" data-id="${p.Rk}">
            <td><input type="checkbox" class="cmp-cb" data-id="${p.Rk}" /></td>
            <td>${i + 1}</td>
            <td>${p.Player}</td>
            <td>${p.Pos}</td>
            <td>${p.Squad}</td>
            <td>${p.Age}</td>
            <td>${(p.similarity_score || 0).toFixed(3)}</td>
          </tr>
        `
            )
            .join("")}
      </tbody>
    </table>`;

    similarTableHolder.querySelectorAll(".srow").forEach((row) => {
        const pid = row.dataset.id;
        row.addEventListener("click", (e) => {
            if (e.target.tagName !== "INPUT") {
                const cb = row.querySelector(".cmp-cb");
                cb.checked = !cb.checked;
            }
            row.classList.toggle(
                "selected-row",
                row.querySelector(".cmp-cb").checked
            );
            togglePulsate(pid);
        });
        row.addEventListener("dblclick", () => {
            doCompareWith(pid);
        });
    });
}

//
// RADAR RENDER
//
function renderRadar(inputRadar, players, inputName) {
    const labels = inputRadar.labels || [];
    const datasets = [];

    // Input dataset (index 0)
    datasets.push({
        label: inputName,
        data: inputRadar.values || [],
        fill: true,
        backgroundColor: "rgba(0,0,0,0)",
        borderColor: "black",
        borderWidth: 3,
        pointRadius: 4,
        _origBackgroundColor: "rgba(0,0,0,0)",
        _origBorderColor: "black",
    });

    datasetMap = {};

    const colorsFill = [
        "rgba(109,172,229,0.45)",
        "rgba(0,99,255,0.45)",
        "rgba(255,99,132,0.45)",
        "rgba(60,179,113,0.45)",
        "rgba(255,159,64,0.45)",
        "rgba(153,102,255,0.45)",
        "rgba(75,192,192,0.45)",
        "rgba(255,205,86,0.45)",
        "rgba(201,203,207,0.45)",
        "rgba(54,162,235,0.45)",
    ];

    const colorsBorder = [
        "rgba(109,172,229,1)",
        "rgba(0,99,255,1)",
        "rgba(255,99,132,1)",
        "rgba(60,179,113,1)",
        "rgba(255,159,64,1)",
        "rgba(153,102,255,1)",
        "rgba(75,192,192,1)",
        "rgba(255,205,86,1)",
        "rgba(201,203,207,1)",
        "rgba(54,162,235,1)",
    ];

    players.forEach((p, i) => {
        const radar = p.radar || { values: [] };
        const bg = colorsFill[i % colorsFill.length];
        const br = colorsBorder[i % colorsBorder.length];
        datasets.push({
            label: p.Player,
            data: radar.values || [],
            fill: true,
            backgroundColor: bg,
            borderColor: br,
            borderWidth: 1.2,
            pointRadius: 3,
            _origBackgroundColor: bg,
            _origBorderColor: br,
            _playerRk: p.Rk,
        });
        datasetMap[p.Rk] = datasets.length - 1;
    });

    const canvas = document.getElementById("radarChartDefault");
    canvas.style.background = "white";
    canvas.style.borderRadius = "10px";
    const ctx = canvas.getContext("2d");

    if (radarChart) radarChart.destroy();

    // external tooltip function
    function externalTooltip(context) {
        const { chart, tooltip } = context;
        const tooltipEl = document.getElementById("customTooltip");
        if (!tooltipEl) return;

        if (tooltip.opacity === 0) {
            if (pointerInsideTooltip || pointerInsideCanvas) {
                return;
            }
            clearTimeout(hideTooltipTimeout);
            hideTooltipTimeout = setTimeout(() => {
                if (!pointerInsideTooltip && !pointerInsideCanvas) {
                    tooltipEl.style.display = "none";
                    tooltipEl.style.opacity = 0;
                }
            }, 180);
            return;
        }

        clearTimeout(hideTooltipTimeout);

        const dataIndex =
            tooltip.dataPoints && tooltip.dataPoints.length
                ? tooltip.dataPoints[0].dataIndex
                : null;
        if (dataIndex === null || dataIndex === undefined) {
            tooltipEl.style.display = "none";
            return;
        }

        const label = chart.data.labels[dataIndex] || "";
        const desc = featureDescriptions[label] || "";

        const rows = chart.data.datasets.map((ds, dsIndex) => {
            const val =
                ds.data && ds.data[dataIndex] !== undefined ? ds.data[dataIndex] : "-";
            const color = ds._origBorderColor || ds.borderColor || "rgba(0,0,0,1)";
            const name = ds.label || `Series ${dsIndex + 1}`;
            return { name, val, color };
        });

        let html = `<div class="tt-title">${label}</div>`;
        if (desc && desc.trim().length > 0) {
            html += `<div class="tt-desc">${desc}</div>`;
            html += `<div style="height:10px;"></div>`;
        }

        html += `<div class="tt-sep"></div>`;
        html += `<div class="tt-rows">`;
        rows.forEach((r) => {
            const displayVal =
                r.val === null || r.val === undefined || r.val === "" ? "-" : r.val;
            html += `
        <div class="tt-row">
          <div class="tt-leftbar" style="background:${r.color};"></div>
          <div class="tt-name">${r.name}</div>
          <div class="tt-val">${displayVal}</div>
        </div>
      `;
        });
        html += `</div>`;

        tooltipEl.innerHTML = html;
        tooltipEl.style.display = "block";
        tooltipEl.style.opacity = 1;

        const canvasRect = chart.canvas.getBoundingClientRect();
        const pageX =
            tooltip.caretX !== undefined
                ? canvasRect.left + tooltip.caretX
                : canvasRect.left + canvasRect.width / 2;
        const pageY =
            tooltip.caretY !== undefined
                ? canvasRect.top + tooltip.caretY
                : canvasRect.top + 20;

        const left = Math.min(pageX + 12, window.innerWidth - 360);
        const top = Math.max(pageY + 10, 8);

        tooltipEl.style.left = `${left}px`;
        tooltipEl.style.top = `${top}px`;
    }

    radarChart = new Chart(ctx, {
        type: "radar",
        data: { labels, datasets },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            animation: { duration: 700, easing: "easeOutBack" },
            plugins: {
                legend: { position: "right", labels: { color: "black" } },
                tooltip: {
                    enabled: false,
                    external: externalTooltip,
                },
            },
            scales: {
                r: {
                    angleLines: { color: "rgba(220,220,220,0.9)", lineWidth: 1.8 },
                    grid: { color: "rgba(220,220,220,0.9)", lineWidth: 1.4 },
                    pointLabels: { color: "black", font: { size: 14 } },
                    ticks: { display: false },
                },
            },
        },
    });

    radarChart.data.datasets.forEach((ds) => {
        if (!ds._origBackgroundColor) ds._origBackgroundColor = ds.backgroundColor;
        if (!ds._origBorderColor) ds._origBorderColor = ds.borderColor;
    });

    const canvasEl = document.getElementById("radarChartDefault");
    canvasEl.addEventListener("mouseenter", () => {
        pointerInsideCanvas = true;
    });
    canvasEl.addEventListener("mouseleave", () => {
        pointerInsideCanvas = false;
        clearTimeout(hideTooltipTimeout);
        hideTooltipTimeout = setTimeout(() => {
            if (!pointerInsideTooltip && !pointerInsideCanvas) {
                customTooltip.style.display = "none";
            }
        }, 180);
    });

    customTooltip.addEventListener("mouseenter", () => {
        pointerInsideTooltip = true;
        clearTimeout(hideTooltipTimeout);
    });
    customTooltip.addEventListener("mouseleave", () => {
        pointerInsideTooltip = false;
        clearTimeout(hideTooltipTimeout);
        hideTooltipTimeout = setTimeout(() => {
            if (!pointerInsideTooltip && !pointerInsideCanvas) {
                customTooltip.style.display = "none";
            }
        }, 180);
    });

    stopPulsate();
}

//
// PULSATE + FADE LOGIC
//
function togglePulsate(playerRk) {
    if (pulsateState.activeId === playerRk) {
        stopPulsate();
        return;
    }
    startPulsate(playerRk);
}

function startPulsate(playerRk) {
    stopPulsate();

    const idx = datasetMap[playerRk];
    if (idx === undefined || !radarChart) return;

    pulsateState.activeId = playerRk;
    pulsateState.startTime = performance.now();

    radarChart.data.datasets.forEach((ds, di) => {
        if (di === 0) {
            ds.backgroundColor = ds._origBackgroundColor;
            ds.borderColor = ds._origBorderColor;
        } else if (di === idx) {
            ds.backgroundColor = ds._origBackgroundColor;
            ds.borderColor = ds._origBorderColor;
            ds.borderWidth = 1.2;
        } else {
            ds.backgroundColor = setAlphaRGBA(
                ds._origBackgroundColor || ds.backgroundColor,
                0.12
            );
            ds.borderColor = setAlphaRGBA(
                ds._origBorderColor || ds.borderColor,
                0.28
            );
        }
    });

    radarChart.update();

    function animate(ts) {
        if (!pulsateState.activeId) return;
        const t = (ts - pulsateState.startTime) / 600;
        const w = 1.2 + Math.abs(Math.sin(t)) * 6;
        const dIdx = datasetMap[pulsateState.activeId];
        if (radarChart && radarChart.data.datasets[dIdx]) {
            radarChart.data.datasets[dIdx].borderWidth = w;
            radarChart.data.datasets[dIdx].pointRadius =
                3 + Math.abs(Math.sin(t)) * 2;
            radarChart.update("none");
        }
        pulsateState.raf = requestAnimationFrame(animate);
    }
    pulsateState.raf = requestAnimationFrame(animate);
}

function stopPulsate() {
    if (pulsateState.raf) cancelAnimationFrame(pulsateState.raf);

    if (radarChart && radarChart.data && radarChart.data.datasets) {
        radarChart.data.datasets.forEach((ds) => {
            if (ds._origBackgroundColor) ds.backgroundColor = ds._origBackgroundColor;
            if (ds._origBorderColor) ds.borderColor = ds._origBorderColor;
            ds.borderWidth = ds._origBorderWidth || ds.borderWidth || 1.2;
            ds.pointRadius = ds._origPointRadius || ds.pointRadius || 3;
        });
        if (radarChart.data.datasets[0])
            radarChart.data.datasets[0].borderWidth = 3;
        radarChart.update();
    }

    pulsateState.activeId = null;
    pulsateState.raf = null;
    pulsateState.startTime = null;
}

//
// COMPARE MULTIPLE / API
//
async function doCompareWith(id) {
    await compareMultiple([selectedPlayerId, id]);
}

compareSelectedBtn.addEventListener("click", async () => {
    const checked = [...document.querySelectorAll(".cmp-cb:checked")].map(
        (cb) => cb.dataset.id
    );
    if (!checked.length) return showError("Select at least one player.");
    await compareMultiple([selectedPlayerId, ...checked]);
});

async function compareMultiple(ids) {
    console.log('[DEBUG] Starting compareMultiple with IDs:', ids);
    try {
        console.log('[DEBUG] Fetching /api/compare_players...');
        const r = await fetch("/api/compare_players", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ player_ids: ids }),
        });
        const j = await r.json();
        console.log('[DEBUG] Received response:', j);
        console.log('[DEBUG] AI Report length:', j.ai_report ? j.ai_report.length : 0);
        if (!j.ok) return showError("Compare failed.");

        lastCompareResponse = j;
        renderCompareBoxes(j);

        renderRadar(
            j.radar[0],
            j.players
                .map((p, i) => (i > 0 ? { ...p, radar: j.radar[i] } : null))
                .filter(Boolean),
            j.players[0].Player
        );

        // Render AI report with error handling
        console.log('[DEBUG] AI Report text:', j.ai_report ? j.ai_report.substring(0, 100) : 'NULL');
        console.log('[DEBUG] marked available?', typeof marked !== 'undefined');
        
        if (j.ai_report && j.ai_report.trim()) {
            try {
                // Always use simple rendering for reliability
                let rendered = j.ai_report;
                
                // Try markdown parsing
                if (typeof marked !== 'undefined' && marked.parse) {
                    console.log('[DEBUG] Using marked.parse()');
                    rendered = marked.parse(j.ai_report);
                } else {
                    console.log('[DEBUG] Using fallback rendering');
                    // Convert markdown-like syntax to HTML manually
                    rendered = rendered
                        .replace(/\n/g, '<br>')
                        .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
                        .replace(/\*(.+?)\*/g, '<em>$1</em>');
                }
                
                aiReportBox.innerHTML = rendered;
                aiReportBox.style.display = "block";
                toggleAiReportBtn.innerText = "Hide AI Report";
                console.log('[DEBUG] AI Report rendered successfully');
            } catch (err) {
                console.error('[DEBUG] Error rendering AI report:', err);
                aiReportBox.innerHTML = '<div style="padding: 20px; background: #fff; color: #333;"><h3>AI Report (Raw)</h3><pre style="white-space: pre-wrap;">' + j.ai_report + '</pre></div>';
                aiReportBox.style.display = "block";
                toggleAiReportBtn.innerText = "Hide AI Report";
            }
        } else {
            console.log('[DEBUG] No AI report in response');
            aiReportBox.innerHTML = '<p style="padding: 20px; color: #ff6b6b;">No AI report was generated. Please try again.</p>';
            aiReportBox.style.display = "block";
            toggleAiReportBtn.innerText = "Hide AI Report";
        }
    } catch (err) {
        showError(err.message || String(err));
    }
}

//
// RENDER COMPARE BOXES
//
function renderCompareBoxes(j) {
    const keys = j.compare_stats.keys || [];
    const rows = j.compare_stats.rows || [];

    const inp = rows[0] || { stats: {} };
    inputPlayerContent.innerHTML = `
    <div class="stat-row"><span class="stat-key">Player</span><span class="stat-value">${j.players[0].Player
        }</span></div>
    ${keys
            .filter((k) => k !== "Player")
            .map(
                (k) => `
      <div class="stat-row">
        <span class="stat-key" data-key="${k}">${k}</span>
        <span class="stat-value">${inp.stats[k] === undefined ? "" : inp.stats[k]
                    }</span>
      </div>
    `
            )
            .join("")}
  `;
    attachBoxTooltips(inputPlayerContent);

    selectedMultipleHolder.innerHTML = rows
        .slice(1)
        .map(
            (row, i) => `
    <div class="compare-box">
      <div class="stat-row"><span class="stat-key">Player</span><span class="stat-value">${j.players[i + 1].Player
                }</span></div>
      ${keys
                    .filter((k) => k !== "Player")
                    .map(
                        (k) => `
        <div class="stat-row">
          <span class="stat-key" data-key="${k}">${k}</span>
          <span class="stat-value">${row.stats[k] === undefined ? "" : row.stats[k]
                            }</span>
        </div>
      `
                    )
                    .join("")}
    </div>
  `
        )
        .join("");

    attachBoxTooltips(selectedMultipleHolder);
}

//
// SHOW ALL FEATURES
//
toggleAllFeaturesBtn.addEventListener("click", () => {
    if (!lastCompareResponse) return showError("Compare players first.");
    showAllFeatures = !showAllFeatures;
    toggleAllFeaturesBtn.innerText = showAllFeatures
        ? "Hide ALL features"
        : "Show ALL features";

    if (showAllFeatures) {
        const players = lastCompareResponse.players || [];
        const p0 = players[0] || {};
        inputPlayerContent.innerHTML =
            `<div class="stat-row"><span class="stat-key">Player</span><span class="stat-value">${p0.Player || ""
            }</span></div>` +
            Object.keys(p0)
                .filter((k) => k !== "Rk")
                .map((k) => {
                    return `<div class="stat-row"><span class="stat-key" data-key="${k}">${k}</span><span class="stat-value">${p0[k] === undefined ? "" : p0[k]
                        }</span></div>`;
                })
                .join("");
        attachBoxTooltips(inputPlayerContent);

        selectedMultipleHolder.innerHTML = players
            .slice(1)
            .map((p) => {
                return `<div class="compare-box">
        <div class="stat-row"><span class="stat-key">Player</span><span class="stat-value">${p.Player || ""
                    }</span></div>
        ${Object.keys(p)
                        .filter((k) => k !== "Rk")
                        .map((k) => {
                            return `<div class="stat-row"><span class="stat-key" data-key="${k}">${k}</span><span class="stat-value">${p[k] === undefined ? "" : p[k]
                                }</span></div>`;
                        })
                        .join("")}
      </div>`;
            })
            .join("");
        attachBoxTooltips(selectedMultipleHolder);
    } else {
        renderCompareBoxes(lastCompareResponse);
    }
});

//
// REMOVE SELECTION
//
removeSelectionBtn.addEventListener("click", () => {
    document.querySelectorAll(".cmp-cb").forEach((cb) => {
        cb.checked = false;
        const row = cb.closest("tr");
        if (row) row.classList.remove("selected-row");
    });

    selectedMultipleHolder.innerHTML =
        'Select players and click "Compare selected"';
    aiReportBox.innerHTML = "";
    aiReportBox.style.display = "none";
    toggleAiReportBtn.innerText = "Show AI Report";

    if (lastSimilarInputRadar && lastSimilarResults) {
        renderRadar(
            lastSimilarInputRadar,
            lastSimilarResults.slice(0, 10),
            selectedPlayerName
        );
    }

    stopPulsate();
});

//
// TOGGLE AI REPORT
//
toggleAiReportBtn.addEventListener("click", () => {
    if (aiReportBox.style.display === "none") {
        aiReportBox.style.display = "block";
        toggleAiReportBtn.innerText = "Hide AI Report";
    } else {
        aiReportBox.style.display = "none";
        toggleAiReportBtn.innerText = "Show AI Report";
    }
});
