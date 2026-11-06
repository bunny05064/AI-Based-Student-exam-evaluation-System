/* Result page: render an evaluation result from sessionStorage or the API. */
(function () {
  "use strict";

  const resultPage = {
    async init() {
      const params = new URLSearchParams(window.location.search);
      const evalId = params.get("evaluation_id");
      let data = null;

      // The POST response is used when available so the page renders without a
      // second request; a refresh falls back to the API, which returns the
      // same shape via app.serializers.serialize_evaluation.
      const cached = sessionStorage.getItem("last_result");
      if (cached) {
        try {
          data = JSON.parse(cached);
        } catch (_) {
          data = null;
        }
        if (data && evalId && data.evaluation_id !== evalId) data = null;
      }
      if (!data && evalId) {
        const r = await app.api("/api/v1/evaluations/" + encodeURIComponent(evalId));
        if (r.ok) data = r.body && r.body.data;
      }
      sessionStorage.removeItem("last_result");

      const loading = app.$("loading-holder");
      if (loading) loading.classList.add("hidden");
      const root = app.$("result-root");

      if (!data) {
        root.innerHTML =
          '<section class="card"><h2>Not found</h2>' +
          '<p class="muted">No evaluation result found for this address.</p>' +
          '<a class="btn btn-primary" href="/evaluate">Evaluate an answer</a></section>';
        return;
      }
      this.render(data);
    },

    render(d) {
      const root = app.$("result-root");
      const c = d.concepts || {};
      const results = Array.isArray(c.results) ? c.results : [];
      const pct = d.percentage || 0;
      const esc = app.escapeHtml;

      const list = (arr, cls) =>
        arr && arr.length
          ? arr.map((n) => `<li class="${cls}">${esc(n)}</li>`).join("")
          : "<li class='muted'>None</li>";

      // Evidence rows show the recognition factor, confidence, rubric weight
      // and score contribution that actually produced the score.
      const evidenceRows = results
        .filter((r) => r.status !== "missing")
        .slice(0, 8)
        .map((r) => {
          const detail = [];
          if (r.factor !== undefined && r.factor !== null) {
            detail.push(`factor ${Number(r.factor).toFixed(2)}`);
          }
          if (r.confidence !== undefined && r.confidence !== null) {
            detail.push(`confidence ${Number(r.confidence).toFixed(2)}`);
          }
          if (r.weight !== undefined && r.weight !== null) {
            detail.push(`weight ${Number(r.weight)}`);
          }
          if (r.contribution !== undefined && r.contribution !== null) {
            detail.push(`contributes ${Number(r.contribution).toFixed(2)}`);
          }
          return (
            `<div class="evidence-row">` +
            (r.evidence ? `<span class="quote">${esc(r.evidence)}</span><br>` : "") +
            `<span class="tag">&rarr; ${esc(r.concept)}</span> ` +
            `<span class="badge badge-${this.badgeClass(r.status)}">${esc(
              this.labelStatus(r.status)
            )}</span>` +
            (detail.length ? ` <span class="muted">(${esc(detail.join(", "))})</span>` : "") +
            `</div>`
          );
        })
        .join("");

      const recs = (d.recommendations || [])
        .map(
          (r) =>
            `<li><span class="tag">${esc(r.kind || r.topic || "tip")}</span> ` +
            `${esc(r.text || "")}</li>`
        )
        .join("");

      const modeBadge =
        d.model_mode === "development"
          ? '<span class="badge badge-dev">Development model mode</span>'
          : "";

      // In development mode the classifier is a no-op, so `label` is null and
      // `model_name` is "none". Saying that is better than rendering an empty
      // box, and better than inventing a label.
      const correctness = d.correctness_prediction && d.correctness_prediction.label
        ? `<p class="muted">Correctness prediction: <strong>${esc(
            d.correctness_prediction.label
          )}</strong> (confidence ${esc(
            d.correctness_prediction.confidence ?? "n/a"
          )}, model ${esc(d.correctness_prediction.model_name || "unknown")})</p>`
        : '<p class="muted">No trained correctness classifier was loaded, so no ' +
          "correctness label is shown.</p>";

      const ocrLine = d.ocr
        ? `(OCR engine: ${esc(d.ocr.engine)}, confidence ${esc(
            d.ocr.confidence
          )}, ${esc(d.ocr.page_count ?? 1)} page(s))`
        : "";

      root.innerHTML = `
        <section class="card">
          <h2>Result ${modeBadge}</h2>
          <div class="score-card">
            <div class="score-ring" style="--pct:${pct}%">
              <div class="score-text">${esc(d.score)} / ${esc(d.max_score)}</div>
              <div class="pct-text">${esc(pct)}%</div>
            </div>
            <div class="score-meta">
              <p><strong>Question:</strong><br>${esc(d.question || "")}</p>
              <p class="muted">Rubric source: ${esc(d.rubric_source || "unknown")}.
                 ${d.source === "image" ? "Answered via handwritten-image OCR. " : ""}
                 ${ocrLine}</p>
              ${d.created_at ? `<p class="muted">Evaluated: ${esc(d.created_at)}</p>` : ""}
              ${correctness}
            </div>
          </div>
        </section>

        <section class="card">
          <div class="grid-3" style="margin:0;padding:0;max-width:none">
            <div>
              <h3>Correct Concepts</h3>
              <ul class="list-check">${list(c.correct, "ok")}</ul>
            </div>
            <div>
              <h3>Partially Correct</h3>
              <ul class="list-check">${list(c.partially_correct, "partial")}</ul>
            </div>
            <div>
              <h3>Missing / Incorrect</h3>
              <ul class="list-check">${list(c.missing, "miss")}${list(
                c.incorrect,
                "miss"
              )}</ul>
            </div>
          </div>
        </section>

        <section class="card">
          <h3>Evidence</h3>
          ${evidenceRows || '<p class="muted">No concept evidence found.</p>'}
        </section>

        <section class="card">
          <h3>Feedback</h3>
          <div class="feedback-block">${esc(d.feedback || "")}</div>
          ${
            d.improvement_tip
              ? `<div class="feedback-block"><strong>How to improve:</strong> ${esc(
                  d.improvement_tip
                )}</div>`
              : ""
          }
        </section>

        <section class="card">
          <h3>Recommendations</h3>
          <ul class="rec-list">${recs || "<li class='muted'>No recommendations.</li>"}</ul>
          <a class="btn btn-outline" href="/evaluate">Evaluate another answer</a>
        </section>
      `;
    },

    badgeClass(s) {
      if (s === "correct") return "correct";
      if (s === "partially_correct") return "partial";
      return "missing";
    },

    labelStatus(s) {
      if (s === "partially_correct") return "partially correct";
      return s || "";
    },
  };

  window.resultPage = resultPage;
})();
