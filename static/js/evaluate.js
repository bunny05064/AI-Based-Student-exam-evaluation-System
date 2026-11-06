/* Evaluate page: text and image evaluation via REST API. */
(function () {
  "use strict";

  const evaluatePage = {
    selectedFile: null,

    init() {
      const input = app.$("image-input");
      if (input) {
        input.addEventListener("change", () => {
          this.selectedFile = input.files && input.files[0] ? input.files[0] : null;
          const chosen = app.$("file-chosen");
          if (chosen) chosen.textContent = this.selectedFile ? this.selectedFile.name : "";
          const clear = app.$("file-clear");
          if (clear) clear.classList.toggle("hidden", !this.selectedFile);
        });
      }

      const clearBtn = app.$("file-clear");
      if (clearBtn) {
        clearBtn.addEventListener("click", () => {
          input.value = "";
          this.selectedFile = null;
          const chosen = app.$("file-chosen");
          if (chosen) chosen.textContent = "";
          clearBtn.classList.add("hidden");
        });
      }

      // Existing students, so a numeric id never has to be guessed.
      app.loadStudentOptions(app.$("student-picker"), {
        onChange: (event) => {
          const nameInput = app.$("student-name");
          const selected = event.target.options[event.target.selectedIndex];
          // Picking a saved student fills the name; the server still matches by
          // name, so an edited name creates a new student rather than
          // silently overwriting someone else's history.
          if (nameInput && selected && selected.value) nameInput.value = selected.textContent;
        },
      });
    },

    /** Parse an optional JSON textarea into an object, or throw a clear error. */
    parseRubric(raw, labelEl) {
      const text = (raw || "").trim();
      if (!text) return null;
      let parsed;
      try {
        parsed = JSON.parse(text);
      } catch (err) {
        throw new Error(labelEl + " must be valid JSON.");
      }
      if (parsed === null || typeof parsed !== "object" || Array.isArray(parsed)) {
        throw new Error(labelEl + " must be a JSON object with a 'concepts' array.");
      }
      if (!Array.isArray(parsed.concepts) || parsed.concepts.length === 0) {
        throw new Error(labelEl + ' needs a non-empty "concepts" array.');
      }
      return parsed;
    },

    async run() {
      const btn = app.$("eval-btn");
      const loading = app.$("loading");
      const errorBanner = app.$("error-banner");

      const question = app.$("question").value.trim();
      const answer = app.$("answer").value.trim();
      const maxScoreRaw = app.$("max-score").value;
      const subject = (app.$("subject") ? app.$("subject").value : "").trim();
      const studentName = app.$("student-name").value.trim();

      errorBanner.classList.add("hidden");
      if (!question) return this.showError("Please enter a question.");
      if (!answer && !this.selectedFile) {
        return this.showError("Please enter a student answer or choose a file.");
      }

      let rubric;
      let rubricJson;
      try {
        rubric = this.parseRubric(app.$("rubric").value, "The rubric");
        rubricJson = app.$("rubric").value.trim();
      } catch (err) {
        return this.showError(err.message);
      }

      const maxScore = parseInt(maxScoreRaw, 10);
      if (!Number.isFinite(maxScore) || maxScore < 1 || maxScore > 100) {
        return this.showError("Max score must be a whole number between 1 and 100.");
      }

      btn.disabled = true;
      loading.classList.remove("hidden");
      try {
        // Resolve the student first so a failure is reported before uploading.
        let studentId = null;
        if (studentName) {
          const resolved = await app.ensureStudent(studentName);
          if (!resolved) {
            return this.showError(
              "Could not save the student name. Please try again."
            );
          }
          studentId = resolved.id;
        }

        let result;
        if (this.selectedFile) {
          const fd = new FormData();
          fd.append("question", question);
          fd.append("image", this.selectedFile);
          fd.append("max_score", String(maxScore));
          if (subject) fd.append("subject", subject);
          if (rubricJson) fd.append("rubric", rubricJson);
          if (studentId) fd.append("student_id", String(studentId));

          const r = await app.api("/api/v1/evaluate/image", {
            method: "POST",
            body: fd,
          });
          if (!r.ok) return this.showError(app.friendlyError(r.status, r.body));
          result = r.body && r.body.data;
        } else {
          const payload = { question, answer, max_score: maxScore };
          if (subject) payload.subject = subject;
          if (rubric) payload.rubric = rubric;
          if (studentId) payload.student_id = studentId;

          const r = await app.api("/api/v1/evaluate", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
          });
          if (!r.ok) return this.showError(app.friendlyError(r.status, r.body));
          result = r.body && r.body.data;
        }

        if (!result) {
          return this.showError("The server returned an empty response. Please try again.");
        }

        sessionStorage.setItem("last_result", JSON.stringify(result));
        window.location.href = "/result?evaluation_id=" + encodeURIComponent(result.evaluation_id || "");
      } catch (e) {
        this.showError("Something went wrong. Please try again.");
      } finally {
        btn.disabled = false;
        loading.classList.add("hidden");
      }
    },

    showError(msg) {
      const error = app.$("error-banner");
      if (!error) return;
      error.textContent = msg;
      error.classList.remove("hidden");
    },
  };

  window.evaluatePage = evaluatePage;
})();
