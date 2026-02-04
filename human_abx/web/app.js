const config = {
  audioBaseUrl: "audio",
  audioBaseUrlSyn: "",
  audioExtension: ".wav",
  isiMs: 300,
  xIsiMs: 450,
  preDelayMs: 400,
  autoPlays: 2,
  // Screen out if catch fail rate exceeds this (checked after minCatchesBeforeScreening)
  catchFailThreshold: 0.35,
  minCatchesBeforeScreening: 2,
  completionCode: "PROLIFIC-CODE",
  submitUrl: "save_responses.php",
  // Demo trial uses hardcoded files (X = A, so correct answer is A)
  demoFileA: "audio/demo_a_UMbrella.wav",
  demoFileB: "audio/demo_b_umBRElla.wav",
};

const prolific = {
  prolificPid: "",
  studyId: "",
  sessionId: "",
};

let isProlific = false;

const state = {
  orderedTrials: [],
  playbackId: 0,
  currentIndex: 0,
  responses: [],
  catchTotal: 0,
  catchFails: 0,
  listId: "--",
  listPath: "",
  participantId: "--",
  started: false,
  loading: false,
  canGoBack: false,
};

const els = {
  introCard: document.getElementById("introCard"),
  taskCard: document.getElementById("taskCard"),
  doneCard: document.getElementById("doneCard"),
  doneRedirectBox: document.getElementById("doneRedirectBox"),
  doneRedirectCount: document.getElementById("doneRedirectCount"),
  doneRedirectBtn: document.getElementById("doneRedirectBtn"),
  doneCompletionRow: document.getElementById("doneCompletionRow"),
  doneError: document.getElementById("doneError"),
  
  failCard: document.getElementById("failCard"),
  failRedirectBox: document.getElementById("failRedirectBox"),
  failRedirectCount: document.getElementById("failRedirectCount"),
  failRedirectBtn: document.getElementById("failRedirectBtn"),
  failCompletionRow: document.getElementById("failCompletionRow"),
  
  errorCard: document.getElementById("errorCard"),
  errorText: document.getElementById("errorText"),
  feedbackOverlay: document.getElementById("feedbackOverlay"),
  feedbackText: document.getElementById("feedbackText"),
  feedbackOk: document.getElementById("feedbackOk"),
  resumeOverlay: document.getElementById("resumeOverlay"),
  resumeYes: document.getElementById("resumeYes"),
  resumeNo: document.getElementById("resumeNo"),
  startButton: document.getElementById("startButton"),
  introHint: document.getElementById("introHint"),
  participantLabel: document.getElementById("participantLabel"),
  listLabel: document.getElementById("listLabel"),
  phaseLabel: document.getElementById("phaseLabel"),
  progressFill: document.getElementById("progressFill"),
  progressText: document.getElementById("progressText"),
  chooseA: document.getElementById("chooseA"),
  chooseB: document.getElementById("chooseB"),
  replayAll: document.getElementById("replayAll"),
  taskHint: document.getElementById("taskHint"),
  completionCode: document.getElementById("completionCode"),
  downloadButton: document.getElementById("downloadButton"),
  clipA: document.getElementById("clipA"),
  clipB: document.getElementById("clipB"),
  clipX: document.getElementById("clipX"),
  stateA: document.getElementById("stateA"),
  stateB: document.getElementById("stateB"),
  stateX: document.getElementById("stateX"),
  audioA: document.getElementById("audioA"),
  audioB: document.getElementById("audioB"),
  audioX: document.getElementById("audioX"),
  // Intro steps
  step1: document.getElementById("step1"),
  step2: document.getElementById("step2"),
  step3: document.getElementById("step3"),
  step4: document.getElementById("step4"),
  step5: document.getElementById("step5"),
  headphonesYes: document.getElementById("headphonesYes"),
  headphonesNo: document.getElementById("headphonesNo"),
  headphonesHint: document.getElementById("headphonesHint"),
  toStep5: document.getElementById("toStep5"),
  goBackRow: document.getElementById("goBackRow"),
  goBackBtn: document.getElementById("goBackBtn"),
  toStep3: document.getElementById("toStep3"),
  demoPlay: document.getElementById("demoPlay"),
  demoChooseA: document.getElementById("demoChooseA"),
  demoChooseB: document.getElementById("demoChooseB"),
  demoChoiceLabel: document.getElementById("demoChoiceLabel"),
  demoHint: document.getElementById("demoHint"),
  demoClipA: document.getElementById("demoClipA"),
  demoClipB: document.getElementById("demoClipB"),
  demoClipX: document.getElementById("demoClipX"),
  demoStateA: document.getElementById("demoStateA"),
  demoStateB: document.getElementById("demoStateB"),
  demoStateX: document.getElementById("demoStateX"),
};

function getParam(name) {
  const params = new URLSearchParams(window.location.search);
  return params.get(name);
}

function applyConfigFromParams() {
  const submitUrl = getParam("submit");
  const list = getParam("list");

  // Skip mode: only play once
  if (getParam("skip")) {
    config.autoPlays = 1;
  }

  // Prolific params
  prolific.prolificPid = getParam("PROLIFIC_PID") || "";
  prolific.studyId = getParam("STUDY_ID") || "";
  prolific.sessionId = getParam("SESSION_ID") || "";
  isProlific = !!prolific.prolificPid;

  if (submitUrl) config.submitUrl = submitUrl;

  // Use PROLIFIC_PID if available, otherwise fall back to participant_id param
  const participantId = prolific.prolificPid || getParam("participant_id") || "";
  if (participantId) {
    state.participantId = participantId;
    els.participantLabel.textContent = `Participant: ${participantId}`;
  }

  if (list) {
    state.listId = list.split("/").pop() || list;
    els.listLabel.textContent = `List: ${state.listId}`;
  }
}

function inferDataset(listUrl) {
  const datasets = ["stress", "pitch_accent", "mandarin_tone"];
  for (const ds of datasets) {
    if (listUrl.includes(`/${ds}/`) || listUrl.startsWith(`${ds}/`)) {
      return ds;
    }
  }
  return null;
}

function setIntroHint(text) {
  els.introHint.textContent = text || "";
}

function setTaskHint(text) {
  els.taskHint.textContent = text || "";
}

function enableStart(enabled) {
  els.startButton.disabled = !enabled;
}

const STORAGE_PREFIX = "human_abx_progress_";

function getStorageKey() {
  if (!state.participantId || !state.listId) return null;
  return `${STORAGE_PREFIX}${state.participantId}_${state.listId}`;
}

function saveProgress() {
  const key = getStorageKey();
  if (!key) return;
  const data = {
    currentIndex: state.currentIndex,
    responses: state.responses,
    catchTotal: state.catchTotal,
    catchFails: state.catchFails,
    timestamp: Date.now(),
  };
  try {
    localStorage.setItem(key, JSON.stringify(data));
  } catch (e) {
    console.warn("Failed to save progress", e);
  }
}

function clearProgress() {
  const key = getStorageKey();
  if (key) localStorage.removeItem(key);
}

function restoreProgress(saved) {
  state.currentIndex = saved.currentIndex || 0;
  state.responses = saved.responses || [];
  state.catchTotal = saved.catchTotal || 0;
  state.catchFails = saved.catchFails || 0;
}

function checkAndPromptResume() {
  const key = getStorageKey();
  if (!key) return;
  const raw = localStorage.getItem(key);
  if (!raw) return;

  try {
    const saved = JSON.parse(raw);
    if (saved && saved.currentIndex > 0 && saved.currentIndex < state.orderedTrials.length) {
      // Found valid progress
      els.resumeOverlay.classList.remove("hidden");

      const handleYes = () => {
        els.resumeOverlay.classList.add("hidden");
        restoreProgress(saved);
        startStudy(); // Jump straight to task
      };

      const handleNo = () => {
        els.resumeOverlay.classList.add("hidden");
        clearProgress();
        // Do nothing, let user proceed with tutorial/start button
      };

      els.resumeYes.onclick = handleYes;
      els.resumeNo.onclick = handleNo;
    }
  } catch (e) {
    console.warn("Failed to parse saved progress", e);
  }
}

function parseCsv(text) {
  const rows = [];
  const lines = text.split(/\r?\n/).filter((line) => line.trim() !== "");
  if (!lines.length) return rows;
  const headers = parseCsvLine(lines[0]);
  for (let i = 1; i < lines.length; i += 1) {
    const cols = parseCsvLine(lines[i]);
    const row = {};
    headers.forEach((h, idx) => {
      row[h] = cols[idx] ?? "";
    });
    rows.push(row);
  }
  return rows;
}

function parseCsvLine(line) {
  const out = [];
  let current = "";
  let inQuotes = false;
  for (let i = 0; i < line.length; i += 1) {
    const ch = line[i];
    if (ch === "\"") {
      if (inQuotes && line[i + 1] === "\"") {
        current += "\"";
        i += 1;
      } else {
        inQuotes = !inQuotes;
      }
    } else if (ch === "," && !inQuotes) {
      out.push(current);
      current = "";
    } else {
      current += ch;
    }
  }
  out.push(current);
  return out;
}

function normalizeTrial(raw) {
  const isCatch = String(raw.is_catch || "").toLowerCase() === "true";
  return {
    phone_sequence: raw.phone_sequence,
    phone_sequence_b: raw.phone_sequence_b,
    accent_a: raw.accent_a,
    accent_b: raw.accent_b,
    speaker_ab: raw.speaker_ab,
    speaker_x: raw.speaker_x,
    file_a: raw.file_a,
    file_b: raw.file_b,
    file_x: raw.file_x,
    is_catch: isCatch,
    correct_answer: raw.correct_answer || "",
    audio_source: raw.audio_source || (isCatch ? "syn" : "main"),
  };
}

function makeAudioUrl(path, source) {
  if (!path) return "";
  const useSyn = source === "syn";
  const baseUrl = useSyn && config.audioBaseUrlSyn ? config.audioBaseUrlSyn : config.audioBaseUrl;
  const base = baseUrl ? baseUrl.replace(/\/$/, "") : "";
  const ext = config.audioExtension || "";
  if (path.endsWith(ext)) {
    return base ? `${base}/${path}` : path;
  }
  return base ? `${base}/${path}${ext}` : `${path}${ext}`;
}

function resetClipStates() {
  els.stateA.textContent = "Ready";
  els.stateB.textContent = "Ready";
  els.stateX.textContent = "Ready";
  els.clipA.classList.remove("active");
  els.clipB.classList.remove("active");
  els.clipX.classList.remove("active");
}

function setClipState(clip, stateEl, text) {
  els.clipA.classList.remove("active");
  els.clipB.classList.remove("active");
  els.clipX.classList.remove("active");
  clip.classList.add("active");
  stateEl.textContent = text;
}

function disableChoices() {
  els.chooseA.disabled = true;
  els.chooseB.disabled = true;
}

function enableChoices() {
  els.chooseA.disabled = false;
  els.chooseB.disabled = false;
}

function setReplayEnabled(enabled) {
  els.replayAll.disabled = !enabled;
}

function updateProgress() {
  const total = state.orderedTrials.length;
  const current = Math.min(state.currentIndex, total);
  const pct = total ? (current / total) * 100 : 0;
  els.progressFill.style.width = `${pct}%`;
  els.progressText.textContent = `${current} / ${total}`;
}

async function playAudio(audioEl, url, clipEl, stateEl, label) {
  if (!url) {
    setTaskHint("Missing audio source. Please refresh and try again.");
    return false;
  }
  audioEl.src = url;
  audioEl.currentTime = 0;
  setClipState(clipEl, stateEl, `Playing ${label}`);
  try {
    await audioEl.play();
  } catch (err) {
    clipEl.classList.remove("active");
    stateEl.textContent = "Blocked";
    setTaskHint("Audio playback failed. Please allow audio for this site and press Replay.");
    return false;
  }
  await new Promise((resolve) => {
    audioEl.onended = () => resolve();
  });
  clipEl.classList.remove("active");
  stateEl.textContent = "Played";
  setTaskHint("");
  return true;
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function stopAllAudio() {
  els.audioA.pause();
  els.audioA.currentTime = 0;
  els.audioA.onended = null;
  els.audioB.pause();
  els.audioB.currentTime = 0;
  els.audioB.onended = null;
  els.audioX.pause();
  els.audioX.currentTime = 0;
  els.audioX.onended = null;
}

function waitForVisible() {
  return new Promise((resolve) => {
    if (!document.hidden) {
      resolve();
      return;
    }
    const handler = () => {
      if (!document.hidden) {
        document.removeEventListener("visibilitychange", handler);
        resolve();
      }
    };
    document.addEventListener("visibilitychange", handler);
  });
}

async function playSequence(trial, repeatsOverride = null) {
  // Cancel any ongoing playback
  stopAllAudio();
  state.playbackId += 1;
  const myPlaybackId = state.playbackId;

  // Wait for tab to be visible before playing audio
  await waitForVisible();
  if (state.playbackId !== myPlaybackId) return;

  disableChoices();
  setReplayEnabled(false);
  resetClipStates();

  const source = trial.audio_source || "main";
  const urlA = makeAudioUrl(trial.file_a, source);
  const urlB = makeAudioUrl(trial.file_b, source);
  const urlX = makeAudioUrl(trial.file_x, source);

  const repeats = Math.max(1, repeatsOverride ?? config.autoPlays);
  for (let i = 0; i < repeats; i += 1) {
    if (config.preDelayMs > 0) {
      await sleep(config.preDelayMs);
      if (state.playbackId !== myPlaybackId) return;
    }
    const okA = await playAudio(els.audioA, urlA, els.clipA, els.stateA, "A");
    if (!okA) {
      setReplayEnabled(true);
      return;
    }
    if (state.playbackId !== myPlaybackId) return;
    await sleep(config.isiMs);
    if (state.playbackId !== myPlaybackId) return;
    const okB = await playAudio(els.audioB, urlB, els.clipB, els.stateB, "B");
    if (!okB) {
      setReplayEnabled(true);
      return;
    }
    if (state.playbackId !== myPlaybackId) return;
    await sleep(config.xIsiMs);
    if (state.playbackId !== myPlaybackId) return;
    const okX = await playAudio(els.audioX, urlX, els.clipX, els.stateX, "X");
    if (!okX) {
      setReplayEnabled(true);
      return;
    }
    if (state.playbackId !== myPlaybackId) return;
    if (i < repeats - 1) {
      await sleep(config.isiMs);
      if (state.playbackId !== myPlaybackId) return;
    }
  }

  setReplayEnabled(true);
  enableChoices();
}

function markResponse(trial, response) {
  const correct = trial.correct_answer ? response === trial.correct_answer : null;
  state.responses.push({
    ...trial,
    participant_id: state.participantId,
    list_id: state.listId,
    trial_index: state.currentIndex,
    timestamp: new Date().toISOString(),
    response,
    correct,
  });
  return correct;
}

function updatePhase() {
  els.phaseLabel.textContent = "In progress";
}

function showCard(card) {
  els.introCard.classList.add("hidden");
  els.taskCard.classList.add("hidden");
  els.doneCard.classList.add("hidden");
  els.failCard.classList.add("hidden");
  els.errorCard.classList.add("hidden");
  card.classList.remove("hidden");
}

function showFeedback(title, message, onOk) {
  const titleEl = document.getElementById("feedbackTitle");
  if (titleEl) titleEl.textContent = title;
  els.feedbackText.textContent = message;
  els.feedbackOverlay.classList.remove("hidden");
  const handler = () => {
    els.feedbackOverlay.classList.add("hidden");
    els.feedbackOk.removeEventListener("click", handler);
    onOk();
  };
  els.feedbackOk.addEventListener("click", handler);
}

function countdownRedirect(url, boxEl, countEl, btnEl) {
  boxEl.classList.remove("hidden");
  let seconds = 5;
  countEl.textContent = seconds;

  let timer = null;

  const go = () => {
    if (timer) clearInterval(timer);
    window.location.href = url;
  };

  btnEl.onclick = go;

  timer = setInterval(() => {
    seconds -= 1;
    countEl.textContent = seconds;
    if (seconds <= 0) {
      go();
    }
  }, 1000);
}

async function finishStudy() {
  clearProgress();
  const result = await submitIfConfigured();

  showCard(els.doneCard);

  if (isProlific) {
    // Update the done card message based on bonus status
    const doneTitle = document.querySelector("#doneCard h2");
    const doneMessage = document.querySelector("#doneCard > p");

    if (result.ok && result.isBonus) {
      if (doneTitle) doneTitle.textContent = "Congratulations!";
      if (doneMessage) {
        doneMessage.innerHTML = "<strong>Great job!</strong> Your high accuracy earned you a bonus payment.";
      }
    }

    if (result.ok && result.completionUrl) {
      // Hide static completion row, show redirect box
      els.doneCompletionRow.classList.add("hidden");
      countdownRedirect(result.completionUrl, els.doneRedirectBox, els.doneRedirectCount, els.doneRedirectBtn);
      return;
    }

    // Prolific fallback: show completion code
    els.doneCompletionRow.classList.remove("hidden");
    els.completionCode.textContent = `Completion code: ${config.completionCode}`;
  } else {
    // Non-Prolific: no completion codes or redirects, just show download
    els.doneCompletionRow.classList.remove("hidden");
    els.completionCode.classList.add("hidden");
  }

  if (!result.ok && config.submitUrl) {
    if (result.error) els.doneError.textContent = `Error: ${result.error}`;
    els.doneError.classList.remove("hidden");
    setTaskHint("Auto-submit failed. Please download your responses.");
  }
}

async function failAttentionCheck() {
  clearProgress();
  // Submit partial responses with attention_failed flag
  const result = await submitIfConfigured(true);

  showCard(els.failCard);

  if (result.ok && result.attentionFailUrl) {
    // Hide static completion row, show redirect box
    els.failCompletionRow.classList.add("hidden");
    countdownRedirect(result.attentionFailUrl, els.failRedirectBox, els.failRedirectCount, els.failRedirectBtn);
    return;
  }
  
  els.failCompletionRow.classList.remove("hidden");

  // No fallback: surface error so the issue is visible.
  if (!result.ok) {
     // We are on the fail card, maybe show error there?
     // For now just logging/alerting isn't great, but the user is already on the fail card.
     // Let's reuse the doneError equivalent if we had one, or just do nothing extra
     // as the fail card says "Task ended".
     console.warn("Failed to submit attention check failure");
  }
}

async function nextTrial() {
  if (state.currentIndex >= state.orderedTrials.length) {
    finishStudy();
    return;
  }

  updatePhase();
  updateProgress();
  const trial = state.orderedTrials[state.currentIndex];
  await playSequence(trial);
}

function handleChoice(response) {
  const trial = state.orderedTrials[state.currentIndex];
  if (!trial) return;
  if (els.chooseA.disabled && els.chooseB.disabled) return;
  disableChoices();

  // Hide go-back when making a new choice
  state.canGoBack = false;
  els.goBackRow.classList.add("hidden");

  const correct = markResponse(trial, response);

  // Track catch trial performance
  if (trial.is_catch) {
    state.catchTotal += 1;
    if (correct === false) {
      state.catchFails += 1;

      // Only screen out on Prolific — in-person participants always finish
      if (isProlific && state.catchTotal >= config.minCatchesBeforeScreening) {
        const failRate = state.catchFails / state.catchTotal;
        if (failRate > config.catchFailThreshold) {
          failAttentionCheck();
          return;
        }
      }
    }
  }

  state.currentIndex += 1;

  // Enable go-back after first trial completes
  if (state.currentIndex > 0) {
    state.canGoBack = true;
    els.goBackRow.classList.remove("hidden");
  }

  saveProgress();
  nextTrial();
}

function goBack() {
  if (!state.canGoBack) return;
  state.canGoBack = false;
  els.goBackRow.classList.add("hidden");

  // Remove the last response
  if (state.responses.length > 0) {
    state.responses.pop();
  }

  // Go back to previous trial
  state.currentIndex = Math.max(0, state.currentIndex - 1);

  saveProgress();
  updatePhase();
  updateProgress();
  const trial = state.orderedTrials[state.currentIndex];
  playSequence(trial);
}

function attachHandlers() {
  els.chooseA.addEventListener("click", () => handleChoice("A"));
  els.chooseB.addEventListener("click", () => handleChoice("B"));
  els.goBackBtn.addEventListener("click", goBack);

  els.replayAll.addEventListener("click", async () => {
    const trial = state.orderedTrials[state.currentIndex];
    if (!trial) return;
    await playSequence(trial, 1);
  });

  els.downloadButton.addEventListener("click", downloadResponses);

  document.addEventListener("keydown", (e) => {
    if (els.taskCard.classList.contains("hidden")) return;
    if (e.key === "1") {
      if (!els.chooseA.disabled) handleChoice("A");
    } else if (e.key === "2") {
      if (!els.chooseB.disabled) handleChoice("B");
    } else if (e.key.toLowerCase() === "r") {
      if (!els.replayAll.disabled) els.replayAll.click();
    }
  });
}

function downloadResponses() {
  if (!state.responses.length) return;
  const headers = Object.keys(state.responses[0]);
  const lines = [headers.join(",")];
  for (const row of state.responses) {
    const line = headers.map((h) => csvEscape(String(row[h] ?? ""))).join(",");
    lines.push(line);
  }
  const blob = new Blob([lines.join("\n")], { type: "text/csv" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `abx_responses_${Date.now()}.csv`;
  a.click();
  URL.revokeObjectURL(url);
}

function csvEscape(value) {
  if (value.includes(",") || value.includes("\"") || value.includes("\n")) {
    return `"${value.replace(/\"/g, "\"\"")}"`;
  }
  return value;
}

async function loadTrialsFromUrl(listUrl) {
  setIntroHint("Loading list...");
  state.loading = true;
  try {
    const res = await fetch(listUrl);
    if (!res.ok) throw new Error(`Failed to load list (${res.status})`);
    const text = await res.text();
    const raw = parseCsv(text);
    state.orderedTrials = raw.map(normalizeTrial);
    enableStart(true);
    setIntroHint("");
    checkAndPromptResume();
  } catch (err) {
    els.errorText.textContent = err.message || "Failed to load list.";
    showCard(els.errorCard);
  } finally {
    state.loading = false;
  }
}

async function submitIfConfigured(screenedOut = false) {
  if (!config.submitUrl) return { ok: false };
  try {
    const res = await fetch(config.submitUrl, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        participant_id: state.participantId,
        list_id: state.listId,
        prolific_pid: prolific.prolificPid,
        study_id: prolific.studyId,
        session_id: prolific.sessionId,
        screened_out: screenedOut,
        responses: state.responses,
      }),
    });
    if (!res.ok) return { ok: false };
    const data = await res.json();
    if (screenedOut && !data.attention_fail_url) {
      return { ok: false, error: "Missing attention_fail_url from server." };
    }
    return {
      ok: true,
      completionUrl: data.completion_url || "",
      attentionFailUrl: data.attention_fail_url || "",
      isBonus: data.is_bonus || false,
      accuracy: data.accuracy || 0,
    };
  } catch (err) {
    return { ok: false, error: err };
  }
}

async function startStudy() {
  if (!state.orderedTrials.length) return;
  state.started = true;
  showCard(els.taskCard);
  updateProgress();
  updatePhase();
  setReplayEnabled(false);
  setTaskHint("");

  await nextTrial();
}

function wireStart() {
  els.startButton.addEventListener("click", () => {
    if (state.loading) return;
    startStudy();
  });
}

function showStep(stepNum) {
  els.step1.classList.add("hidden");
  els.step2.classList.add("hidden");
  els.step3.classList.add("hidden");
  els.step4.classList.add("hidden");
  els.step5.classList.add("hidden");
  const step = document.getElementById(`step${stepNum}`);
  if (step) step.classList.remove("hidden");
}

function wireIntroSteps() {
  els.headphonesYes.addEventListener("click", () => {
    els.headphonesHint.textContent = "";
    showStep(2);
  });

  els.headphonesNo.addEventListener("click", () => {
    els.headphonesHint.textContent = "Please put on wired headphones and find a quiet spot before continuing.";
  });

  els.toStep3.addEventListener("click", () => {
    showStep(3);
    // Demo uses hardcoded files, so always ready
    els.demoPlay.disabled = false;
  });

  let demoFirstPlay = true;

  async function playDemoClip(audioEl, url, clipEl, stateEl) {
    clipEl.classList.add("active");
    stateEl.textContent = "Playing...";
    audioEl.src = url;
    audioEl.currentTime = 0;
    try {
      await audioEl.play();
    } catch (err) {
      clipEl.classList.remove("active");
      stateEl.textContent = "Blocked";
      els.demoHint.textContent =
        "Audio playback failed. Please allow audio for this site and try again.";
      return false;
    }
    await new Promise((r) => (audioEl.onended = r));
    stateEl.textContent = "Played";
    clipEl.classList.remove("active");
    return true;
  }

  async function playDemoSequence(urlA, urlB, urlX) {
    // Cancel any ongoing playback
    stopAllAudio();

    // Wait for tab to be visible before playing audio
    await waitForVisible();

    // Reset states
    els.demoClipA.classList.remove("active");
    els.demoClipB.classList.remove("active");
    els.demoClipX.classList.remove("active");
    els.demoStateA.textContent = "Ready";
    els.demoStateB.textContent = "Ready";
    els.demoStateX.textContent = "Ready";

    await sleep(config.preDelayMs);

    // Play A
    const okA = await playDemoClip(els.audioA, urlA, els.demoClipA, els.demoStateA);
    if (!okA) return false;

    await sleep(config.isiMs);

    // Play B
    const okB = await playDemoClip(els.audioB, urlB, els.demoClipB, els.demoStateB);
    if (!okB) return false;

    await sleep(config.isiMs);

    // Play X
    const okX = await playDemoClip(els.audioX, urlX, els.demoClipX, els.demoStateX);
    if (!okX) return false;
    return true;
  }

  els.demoPlay.addEventListener("click", async () => {
    els.demoPlay.disabled = true;
    els.demoChooseA.disabled = true;
    els.demoChooseB.disabled = true;
    els.demoChoiceLabel.style.visibility = "hidden";
    els.demoHint.textContent = "";

    // Use hardcoded demo files (X = A)
    const urlA = config.demoFileA;
    const urlB = config.demoFileB;
    const urlX = config.demoFileA; // X = A

    // Play twice on first play, once on replay
    const repeats = demoFirstPlay ? config.autoPlays : 1;
    let ok = true;
    for (let i = 0; i < repeats; i++) {
      ok = await playDemoSequence(urlA, urlB, urlX);
      if (!ok) break;
      if (i < repeats - 1) {
        await sleep(config.isiMs);
      }
    }

    if (!ok) {
      els.demoPlay.disabled = false;
      return;
    }

    demoFirstPlay = false;

    // Enable choices
    els.demoChoiceLabel.style.visibility = "visible";
    els.demoChooseA.disabled = false;
    els.demoChooseB.disabled = false;
    els.demoPlay.disabled = false;
    els.demoPlay.innerHTML = 'Replay A &rarr; B &rarr; X';
    els.demoHint.textContent = "Now choose: does X sound more like A or B?";
  });

  const handleDemoChoice = (choice) => {
    // Demo correct answer is always A (since X = A)
    const correct = choice === "A";

    els.demoChooseA.disabled = true;
    els.demoChooseB.disabled = true;

    if (correct) {
      els.demoHint.textContent = "Correct! You've got it.";
      setTimeout(() => {
        showStep(4);
      }, 800);
    } else {
      showFeedback(
        "Not quite",
        "Listen carefully to A and B, then decide which one X sounds closer to. Let's try again.",
        () => {
          els.demoPlay.disabled = false;
          els.demoHint.textContent = "";
        }
      );
    }
  };

  els.demoChooseA.addEventListener("click", () => handleDemoChoice("A"));
  els.demoChooseB.addEventListener("click", () => handleDemoChoice("B"));

  els.toStep5.addEventListener("click", () => {
    showStep(5);
  });
}

function setListFromParams() {
  const listUrl = getParam("list");
  const dataset = getParam("dataset");

  // If no list but dataset provided, redirect to assign.php
  if (!listUrl && dataset) {
    const params = new URLSearchParams(window.location.search);
    params.set("dataset", dataset);
    window.location.href = `assign.php?${params.toString()}`;
    return;
  }

  if (!listUrl) {
    throw new Error("Missing required parameter: list (or dataset for auto-assignment)");
  }
  state.listPath = listUrl;
  const inferredDataset = inferDataset(listUrl);
  if (!inferredDataset) {
    throw new Error("Unable to infer dataset from list path.");
  }
  config.audioBaseUrl = `audio/${inferredDataset}`;
  config.audioBaseUrlSyn = `audio/${inferredDataset}_syn`;
  state.listId = listUrl.split("/").pop() || listUrl;
  els.listLabel.textContent = `List: ${state.listId}`;
  loadTrialsFromUrl(listUrl);
}

window.addEventListener("load", () => {
  try {
    applyConfigFromParams();
    if (!config.audioBaseUrl) {
      throw new Error("Missing required parameter: audioBase");
    }
    attachHandlers();
    wireStart();
    wireIntroSteps();
    setListFromParams();

    if (isProlific) {
      els.completionCode.textContent = `Completion code: ${config.completionCode}`;
    } else {
      // Hide Prolific-specific UI elements
      const bonusHint = document.getElementById("bonusHint");
      if (bonusHint) bonusHint.classList.add("hidden");
    }

    // Skip tutorial if requested via URL param
    if (getParam("skip")) {
      const waitForTrials = setInterval(() => {
        if (state.orderedTrials.length > 0 && !document.hidden) {
          clearInterval(waitForTrials);
          startStudy();
        }
      }, 100);
    }
  } catch (err) {
    els.errorText.textContent = err.message || "Unable to start.";
    showCard(els.errorCard);
  }
});
