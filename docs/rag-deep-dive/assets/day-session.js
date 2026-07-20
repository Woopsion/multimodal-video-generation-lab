(() => {
  "use strict";

  const storageKey = document.body.dataset.storageKey || "rag-day-session";
  const steps = [...document.querySelectorAll(".day-step")];
  const stepButtons = [...document.querySelectorAll("[data-step-target]")];
  const progressFill = document.querySelector("#courseProgress");
  const progressText = document.querySelector("#courseProgressText");
  const notes = document.querySelector("#dayNotes");
  const completionNote = document.querySelector("#completionNote");
  const completeButton = document.querySelector("#completeDay");
  const memoryStore = {};

  let state = loadState();
  state.step = Math.min(Math.max(Number(state.step) || 0, 0), steps.length - 1);
  state.maxStep = Math.min(Math.max(Number(state.maxStep) || 0, state.step), steps.length - 1);


  document.querySelectorAll("[data-next-step]").forEach(button => {
    button.addEventListener("click", () => {
      const next = Math.min(state.step + 1, steps.length - 1);
      state.maxStep = Math.max(state.maxStep, next);
      state.step = next;
      saveState();
      renderStep(true);
    });
  });

  document.querySelectorAll("[data-previous-step]").forEach(button => {
    button.addEventListener("click", () => {
      state.step = Math.max(state.step - 1, 0);
      saveState();
      renderStep(true);
    });
  });

  stepButtons.forEach(button => {
    button.addEventListener("click", () => {
      const target = Number(button.dataset.stepTarget);
      if (target > state.maxStep) return;
      state.step = target;
      saveState();
      renderStep(true);
    });
  });

  document.querySelectorAll("[data-case-target]").forEach(button => {
    button.addEventListener("click", () => {
      const target = button.dataset.caseTarget;
      document.querySelectorAll("[data-case-target]").forEach(candidate => {
        candidate.classList.toggle("is-active", candidate === button);
      });
      document.querySelectorAll(".case-panel").forEach(panel => {
        panel.hidden = panel.id !== target;
      });
      state.caseTarget = target;
      saveState();
    });
  });

  document.querySelectorAll("[data-trace-mode]").forEach(button => {
    button.addEventListener("click", () => {
      const mode = button.dataset.traceMode;
      const lab = document.querySelector(".trace-lab");
      lab.dataset.mode = mode;
      document.querySelectorAll("[data-trace-mode]").forEach(candidate => {
        candidate.classList.toggle("is-active", candidate === button);
      });
      document.querySelectorAll(".trace-explanation").forEach(panel => {
        panel.hidden = panel.dataset.traceExplanation !== mode;
      });
      state.traceMode = mode;
      saveState();
    });
  });

  if (notes) {
    notes.addEventListener("input", event => {
      state.notes = event.target.value;
      saveState();
    });
  }

  document.querySelectorAll("[data-confidence]").forEach(button => {
    button.addEventListener("click", () => {
      state.confidence = button.dataset.confidence;
      document.querySelectorAll("[data-confidence]").forEach(candidate => {
        candidate.classList.toggle("is-active", candidate === button);
      });
      saveState();
    });
  });

  if (completeButton) {
    completeButton.addEventListener("click", () => {
      state.done = true;
      state.maxStep = steps.length - 1;
      saveState();
      renderCompletion();
    });
  }

  renderStep();
  restoreControls();

  function renderStep(shouldScroll = false) {
    steps.forEach((step, index) => {
      step.hidden = index !== state.step;
    });

    stepButtons.forEach((button, index) => {
      button.disabled = index > state.maxStep;
      button.classList.toggle("is-active", index === state.step);
    });

    const percent = Math.round(((state.step + 1) / steps.length) * 100);
    progressFill.style.width = percent + "%";
    progressText.textContent = percent + "%";

    document.querySelectorAll("[data-previous-step]").forEach(button => {
      button.disabled = state.step === 0;
    });

    if (shouldScroll) {
      document.querySelector(".day-intro").scrollIntoView({ behavior: "smooth", block: "start" });
    }
  }

  function restoreControls() {
    if (notes) notes.value = state.notes || "";

    const caseTarget = state.caseTarget || "case-rag";
    const caseButton = document.querySelector('[data-case-target="' + caseTarget + '"]');
    if (caseButton) caseButton.click();

    const traceMode = state.traceMode || "retrieval";
    const traceButton = document.querySelector('[data-trace-mode="' + traceMode + '"]');
    if (traceButton) traceButton.click();

    if (state.confidence) {
      const confidenceButton = document.querySelector('[data-confidence="' + state.confidence + '"]');
      if (confidenceButton) confidenceButton.classList.add("is-active");
    }

    renderCompletion();
  }

  function renderCompletion() {
    if (!completionNote || !completeButton) return;
    const dayLabel = document.body.dataset.dayLabel || "Day 01";
    completionNote.hidden = !state.done;
    completeButton.textContent = state.done
      ? "\u5df2\u8bb0\u5f55 " + dayLabel + " \u5b66\u4e60\u7ed3\u679c"
      : "\u8bb0\u5f55 " + dayLabel + " \u5b66\u4e60\u7ed3\u679c";
  }

  function defaultState() {
    return {
      step: Number(document.body.dataset.initialStep) || 0,
      maxStep: Number(document.body.dataset.initialMaxStep) || 0,
      caseTarget: "case-rag",
      traceMode: "retrieval",
      notes: notes ? notes.defaultValue.trim() : "",
      confidence: document.body.dataset.initialConfidence || "",
      done: document.body.dataset.initialDone === "true"
    };
  }

  function loadState() {
    try {
      const saved = window.localStorage.getItem(storageKey);
      return saved ? { ...defaultState(), ...JSON.parse(saved) } : defaultState();
    } catch (error) {
      return { ...defaultState(), ...memoryStore };
    }
  }

  function saveState() {
    try {
      window.localStorage.setItem(storageKey, JSON.stringify(state));
    } catch (error) {
      Object.assign(memoryStore, state);
    }
  }
})();
