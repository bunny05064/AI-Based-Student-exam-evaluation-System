/* Dashboard page: student progress via /api/v1/student/<id>/progress. */
(function () {
  "use strict";

  const dashboardPage = {
    init() {
      this.loadStudentOptions();
    },

    async loadStudentOptions() {
      const picker = app.$("student-picker");
      if (!picker) return;
      await app.loadStudentOptions(picker, {
        onChange: (event) => {
          const idInput = app.$("student-id");
          if (idInput) idInput.value = event.target.value;
          if (event.target.value) this.load();
        },
      });
    },

    async load() {
      const idInput = app.$("student-id");
      // No silent `|| 1` fallback: that silently showed another student's data.
      const studentId = parseInt(idInput.value, 10);
      const error = app.$("error-banner");
      error.classList.add("hidden");

      if (!Number.isFinite(studentId) || studentId < 1) {
        error.textContent = "Please choose a student or enter a valid student id.";
        error.classList.remove("hidden");
        return;
      }

      const r = await app.api("/api/v1/student/" + studentId + "/progress");
      if (!r.ok) {
        error.textContent = app.friendlyError(r.status, r.body);
        error.classList.remove("hidden");
        return;
      }
      this.render(r.body.data);
    },

    render(d) {
      const esc = app.escapeHtml;

      const tag = (items, cls) =>
        items && items.length
          ? items.map((n) => `<span class="tag ${cls}">${esc(n)}</span>`).join(" ")
          : '<span class="muted">None</span>';

      const masteryBars = (d.concept_mastery || [])
        .map((m) => {
          const pct = Math.round((m.mastery || 0) * 100);
          const cls = pct >= 70 ? "strong" : pct < 50 ? "weak" : "developing";
          const width = Math.max(6, Math.min(100, pct));
          return (
            `<div class="mastery-bar-wrap">` +
            `<div class="mastery-label">` +
            `<span>${esc(m.concept)}</span>` +
            `<span>${pct}% &middot; ${esc(m.attempts ?? 0)} attempts</span>` +
            `</div>` +
            `<div class="mastery-bar"><div class="fill fill-${cls}" style="width:${width}%"></div></div>` +
            `</div>`
          );
        })
        .join("");

      const recent = (d.recent_evaluations || [])
        .map(
          (e) =>
            `<div class="evidence-row">` +
            `<span class="quote">${esc(e.question || "")}</span>` +
            `<span class="tag" style="float:right">${esc(e.score)} / ${esc(
              e.max_score
            )} (${esc(e.percentage)}%)</span>` +
            (e.created_at
              ? `<br><span class="muted">${esc(e.created_at)}</span>`
              : "") +
            `</div>`
        )
        .join("");

      const studentName = d.student ? d.student.name : "";

      app.$("dashboard-root").innerHTML = `
        <section class="card">
          <h3>${esc(studentName || "Student")}</h3>
          <div class="kpi-row">
            <div class="kpi"><div class="value">${esc(d.average_score)}%</div><div class="label">Average Score</div></div>
            <div class="kpi"><div class="value">${esc(d.total_evaluations)}</div><div class="label">Evaluations</div></div>
            <div class="kpi"><div class="value">${esc(
              (d.strong_concepts || []).length
            )}</div><div class="label">Strong Concepts</div></div>
            <div class="kpi"><div class="value">${esc(
              (d.weak_concepts || []).length
            )}</div><div class="label">Weak Concepts</div></div>
          </div>
        </section>

        <section class="card">
          <h3>Concept Mastery</h3>
          ${
            masteryBars ||
            '<p class="muted">No concepts tracked yet - evaluate answers for this student.</p>'
          }
        </section>

        <section class="card">
          <h3>Strong Concepts</h3>
          <div class="tag-list">${tag(d.strong_concepts, "tag-strong")}</div>
        </section>

        <section class="card">
          <h3>Developing Concepts</h3>
          <div class="tag-list">${tag(d.developing_concepts, "tag-dev")}</div>
        </section>

        <section class="card">
          <h3>Weak Concepts</h3>
          <div class="tag-list">${tag(d.weak_concepts, "tag-weak")}</div>
        </section>

        <section class="card">
          <h3>Recent Evaluations</h3>
          ${recent || '<p class="muted">No evaluations yet.</p>'}
        </section>
      `;
    },
  };

  window.dashboardPage = dashboardPage;
})();
