/* Shared helpers for the vanilla-JS pages. */
(function () {
  "use strict";

  const app = {
    /**
     * Fetch JSON and never throw.
     *
     * A network failure or an unreachable server used to reject, which left
     * callers with an unhandled promise rejection instead of a message they
     * could show. `status` is 0 for "never reached the server" and the HTTP
     * status otherwise; `body` is null when the response was not JSON.
     */
    async api(path, options) {
      let res;
      try {
        res = await fetch(path, options);
      } catch (_) {
        return { ok: false, status: 0, body: null, networkError: true };
      }
      let body = null;
      try {
        body = await res.json();
      } catch (_) {
        /* non-JSON or empty body */
      }
      return { ok: res.ok, status: res.status, body, networkError: false };
    },

    /**
     * Map backend error codes to student-safe messages (never leak internals).
     */
    friendlyError(status, body) {
      const error = body && body.error;
      const code = error && error.code;
      const message = error && error.message;

      switch (code) {
        case "INVALID_INPUT":
          if (message && /question/i.test(message))
            return "Please enter a valid question.";
          if (message && /answer/i.test(message))
            return "Please enter a student answer before evaluating.";
          return message || "Please check the information you entered.";
        case "VALIDATION_ERROR":
          return message || "Please check the information you entered.";
        case "UNSUPPORTED_FILE":
          return "That file type is not supported. Use JPG, PNG, GIF, BMP, WebP or PDF.";
        case "FILE_TOO_LARGE":
          return "The file is too large. Maximum upload size is 10 MB.";
        case "OCR_FAILED":
          return "We could not read the answer image. Try a clearer photo or type the answer instead.";
        case "RESOURCE_NOT_FOUND":
          return message || "The requested item was not found.";
        case "MODEL_LOAD_FAILED":
        case "INVALID_MODEL_STATE":
          return "The AI model is still warming up. Please try again shortly.";
        case "DATASET_ERROR":
          return message || "The dataset could not be processed.";
        default:
          // `status === 0` means fetch never completed: a network/CORS problem.
          // Anything else is a real server response we should not describe.
          if (!status) {
            return "Network error - please check your connection and try again.";
          }
          return "Something went wrong. Please try again.";
      }
    },

    /** Escape text before interpolating it into innerHTML. */
    escapeHtml(value) {
      if (value === null || value === undefined) return "";
      return String(value)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#39;");
    },

    /** Escape a value used inside an HTML attribute. */
    escapeAttr(value) {
      return app.escapeHtml(value);
    },

    /**
     * Resolve a student id by name, creating the student on first use.
     * Returns null when the request failed so callers can surface the error
     * instead of silently evaluating an unattributed answer.
     */
    async ensureStudent(name) {
      const trimmed = (name || "").trim();
      if (!trimmed) return null;
      const { ok, status, body } = await app.api("/api/v1/student", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: trimmed }),
      });
      if (!ok || !body || !body.data) {
        return null;
      }
      // Surface the reason for the caller; status is carried for messaging.
      return { id: body.data.id, name: body.data.name, status, error: null };
    },

    /** Populate a <select> with recent students so ids are never guessed. */
    async loadStudentOptions(selectEl, { onChange } = {}) {
      if (!selectEl) return;
      const { ok, body } = await app.api("/api/v1/students");
      const students = ok && body && Array.isArray(body.data) ? body.data : [];

      selectEl.innerHTML = '<option value="">New student (enter name)</option>';
      students.forEach((s) => {
        const opt = document.createElement("option");
        opt.value = String(s.id);
        opt.textContent = s.name;
        selectEl.appendChild(opt);
      });

      if (typeof onChange === "function") {
        selectEl.addEventListener("change", onChange);
      }
    },

    async showModelMode() {
      try {
        const { ok, body } = await app.api("/api/v1/model/info");
        if (!ok || !body || !body.data) return;
        const note = document.getElementById("mode-note");
        if (!note) return;
        const mode = body.data.model_mode || "development";
        if (mode === "development") {
          note.innerHTML =
            '<span class="badge badge-dev">Development model mode</span> ' +
            "Running on a classical TF-IDF embedding pipeline (no trained weights). " +
            "Install requirements-ml.txt and set MODEL_MODE=production for transformer evaluation.";
        } else {
          const provider = app.escapeHtml(
            body.data.embedding_provider || "transformers"
          );
          note.innerHTML =
            '<span class="badge badge-correct">Production mode</span> ' +
            `Embedding provider: ${provider}.`;
        }
      } catch (_) {
        /* offline - ignore */
      }
    },

    $(id) {
      return document.getElementById(id);
    },
  };

  window.app = app;
})();
