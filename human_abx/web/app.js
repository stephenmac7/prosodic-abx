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
};

const DEMO_FILES = {
  stress: { a: "demo_a_UMbrella.wav", b: "demo_b_umBRElla.wav" },
  pitch_accent: { a: "demo_a_ichien0.wav", b: "demo_b_ichien2.wav" },
  tone: { a: "demo_a_zhua1.wav", b: "demo_b_zhua3.wav" },
};

const STRINGS = {
  tone: {
    autoSubmitFailed: "自动提交失败。请下载你的回答。", // machine-translated
    beginTask: "任务开始",
    bonusMessage: "<strong>太棒了！</strong> 你的高准确率为你赢得了额外奖励。", // machine-translated
    bonusTitle: "恭喜！", // machine-translated
    clipBlocked: "已拦截", // machine-translated
    clipPlayed: "播放完毕", // machine-translated
    clipPlaying: "正在播放 {label}", // machine-translated
    clipReady: "准备就绪",
    completionCodeLabel: "完成代码", // machine-translated
    demoCorrect: "正确！你已经掌握了。", // machine-translated
    demoPrompt: "现在请选择：X 听起来更像 A 还是 B？", // machine-translated
    demoWrongMsg: "请仔细听 A 和 B，然后决定 X 听起来更像哪一个。让我们再试一次。", // machine-translated
    demoWrongTitle: "不完全正确", // machine-translated
    doneMessage: "感谢参加本次任务",
    doneRedirecting: "正在重定向...", // machine-translated
    doneTitle: "全部完成",
    downloadResponses: "下载回答", // machine-translated
    errorMissingParams: "缺少必要参数。", // machine-translated
    errorPrefix: "错误", // machine-translated
    errorTitle: "错误", // machine-translated
    failMessage: "感谢参加，很遗憾你好像没有认真完成任务",
    failRedirecting: "正在重定向...", // machine-translated
    failReturn: "返回", // machine-translated
    failTitle: "任务结束", // machine-translated
    failedToLoadList: "加载列表失败。", // machine-translated
    failedToLoadListStatus: "加载列表失败 ({status})", // machine-translated
    feedbackOk: "确定", // machine-translated
    feedbackTitle: "反馈", // machine-translated
    goBackBtn: "&larr; 返回上一条",
    gotIt: "我明白了",
    headphonesNo: "否",
    headphonesQuestion: '你现在所处的环境是否安静？是否佩戴<strong>有线</strong>耳机？',
    headphonesSubQuestion: "请避免使用无线耳机（如AirPods等）。无线耳机可能会导致音频截断以及时序问题。",
    headphonesWarning: "在继续之前，请戴上有线耳机并找一个安静的地方。", // machine-translated
    headphonesYes: "是的，我准备好了",
    hintBonus: "<strong>高准确率奖励！</strong><br>准确率高的参与者将获得额外奖励。请保持专注！", // machine-translated
    hintGoBack: '<strong>改主意了？</strong><br>你可以根据需要多次返回上一个问题并修改你的答案',
    hintKeyboard: '<strong>快捷键</strong><br>按 <span class="key">1</span> 选择 A，或者 <span class="key">2</span> 选择 B。',
    hintReplay: '<strong>随时重新播放</strong><br>你可以根据需要随时多次重新播放 A &rarr; B &rarr; X。请按 <span class="key">R</span> 或点击"重新播放"按钮',
    inProgress: "进行中", // machine-translated
    labTaskTitle: "语音辨别任务", // machine-translated
    listLabel: "列表", // machine-translated
    loadingList: "正在加载列表...", // machine-translated
    missingAttentionFailUrl: "服务器未返回 attention_fail_url。", // machine-translated
    missingAudioBase: "错误：缺少 audioBase 参数。", // machine-translated
    missingAudioSource: "缺少音频源。请刷新并重试。", // machine-translated
    missingListParam: "错误：缺少 list 参数。", // machine-translated
    pageTitle: "语音辨别任务", // machine-translated
    participantLabel: "参与者", // machine-translated
    playDemo: "播放 A &rarr; B &rarr; X",
    playbackFailedReplay: "音频播放失败。请允许本网站播放音频并点击“重新播放”。", // machine-translated
    playbackFailedTryAgain: "音频播放失败。请允许本网站播放音频并重试。", // machine-translated
    redirectContinue: "继续", // machine-translated
    replayBtn: '重新播放 A &rarr; B &rarr; X <span class="key">R</span>',
    replayDemo: "重新播放 A &rarr; B &rarr; X", // machine-translated
    reportProblem: "报告问题", // machine-translated
    resumeMessage: "发现了已保存的进度。", // machine-translated
    resumeNo: "重新开始",
    resumeTitle: "恢复进度？", // machine-translated
    resumeYes: "从上次离开处继续",
    step1Intro: "在本任务中，你将听到一些汉语音节的发音，并需要根据听到的内容做出判断。开始之前，请确认以下事项：",
    step1Title: "语音辨别任务",
    step2Clips: '你将依次听到三个汉语普通话的音节录音：<strong>A</strong>、<strong>B</strong>、和 <strong>X</strong>',
    step2Note: '注意: <strong>A</strong> 和 <strong>B</strong> 是不同的录音.',
    step2Task: '你的任务：判断音节<strong>X</strong>听起来更像音节<strong>A</strong>还是音节<strong>B</strong>',
    step2Title: "任务说明",
    step3Intro: "请先听下方的 A、B 和 X 的音频，再选择 X 听起来更像 A 还是 B。",
    step3Title: "任务练习",
    step4Title: "提示",
    step5Intro: "非常好！你已经可以开始正式的语音辨别任务了",
    step5Title: "你准备好了！",
    unableToInferDataset: "无法从路径推断数据集。", // machine-translated
    unableToStart: "无法开始任务。", // machine-translated
    xSoundsLike: "X 听起来更像...",
  },
  pitch_accent: {
    autoSubmitFailed: "自動送信に失敗しました。回答をダウンロードしてください。",
    beginTask: "タスクを開始する",
    bonusMessage: "<strong>素晴らしい！</strong> 高い正解率を達成したため、ボーナス報酬が確定しました。",
    bonusTitle: "おめでとうございます！",
    clipBlocked: "ブロックされました",
    clipPlayed: "再生済み",
    clipPlaying: "{label} を再生中",
    clipReady: "準備完了",
    completionCodeLabel: "完了コード",
    demoCorrect: "正解です！その調子です。",
    demoPrompt: "X は A と B のどちらに近いと感じましたか？",
    demoWrongMsg: "A と B をよく聞き比べて、X がどちらに近いかもう一度判断してみてください。",
    demoWrongTitle: "惜しい！",
    doneMessage: "タスクにご協力いただきありがとうございました。",
    doneRedirecting: "リダイレクト中...",
    doneTitle: "完了",
    downloadResponses: "回答をダウンロード",
    errorMissingParams: "必要なパラメータが不足しています。",
    errorPrefix: "エラー",
    errorTitle: "エラー",
    failMessage: "ご協力ありがとうございました。誠に残念ながら、タスクが適切に完了されなかったようです。",
    failRedirecting: "リダイレクト中...",
    failReturn: "戻る",
    failTitle: "タスク終了",
    failedToLoadList: "リストの読み込みに失敗しました。",
    failedToLoadListStatus: "リストの読み込みに失敗しました ({status})",
    feedbackOk: "OK",
    feedbackTitle: "フィードバック",
    goBackBtn: "&larr; 前の問題に戻る",
    gotIt: "了解しました",
    headphonesNo: "いいえ",
    headphonesQuestion: '周囲の環境は静かですか？また、<strong>有線</strong>ヘッドホンまたはイヤホンを着用していますか？',
    headphonesSubQuestion: "ワイヤレスイヤホン（AirPodsなど）の使用は避けてください。音声の途切れやタイミングの問題が発生する可能性があります。",
    headphonesWarning: "有線ヘッドホンを着用し、静かな場所へ移動してから続けてください。",
    headphonesYes: "はい、準備ができました",
    hintBonus: '<strong>ボーナスについて</strong><br>正解率が高い方には、追加のボーナス報酬が支払われます！集中して取り組んでください。',
    hintGoBack: '<strong>答えを直したい場合</strong><br>必要に応じて、前の問題に戻って回答を修正することができます。',
    hintKeyboard: '<strong>ショートカットキー</strong><br><span class="key">1</span> キーで A を、<span class="key">2</span> キーで B を選択できます。',
    hintReplay: '<strong>いつでも再再生可能</strong><br>必要に応じて、何度でも A &rarr; B &rarr; X を再生できます。<span class="key">R</span> キーを押すか、「もう一度再生」ボタンをクリックしてください。',
    inProgress: "進行中",
    labTaskTitle: "音声判別タスク",
    listLabel: "リスト番号",
    loadingList: "リストを読み込み中...",
    missingAttentionFailUrl: "サーバーから attention_fail_url が返されませんでした。",
    missingAudioBase: "エラー：audioBase が設定されていません。",
    missingAudioSource: "音声ソースが見つかりません。ページを更新してもう一度お試しください。",
    missingListParam: "エラー：list パラメータが必要です。",
    pageTitle: "音声判別タスク",
    participantLabel: "参加者ID",
    playDemo: "A &rarr; B &rarr; X を再生",
    playbackFailedReplay: "再生に失敗しました。音声再生を許可してから「もう一度再生」を押してください。",
    playbackFailedTryAgain: "再生に失敗しました。このサイトでの音声再生を許可し、もう一度お試しください。",
    redirectContinue: "続行",
    replayBtn: 'A &rarr; B &rarr; X をもう一度再生 <span class="key">R</span>',
    replayDemo: "A &rarr; B &rarr; X をもう一度再生",
    reportProblem: "問題を報告",
    resumeMessage: "保存された進行状況が見つかりました。",
    resumeNo: "最初からやり直す",
    resumeTitle: "再開しますか？",
    resumeYes: "前回の続きから再開",
    step1Intro: "このタスクでは、日本語の録音を聞き、その内容に基づいて判断を行っていただきます。開始前に、以下の点を確認してください：",
    step1Title: "音声判別タスク",
    step2Clips: '日本語の単語の録音が3つ、順番に流れます：<strong>A</strong>、<strong>B</strong>、そして <strong>X</strong>',
    step2Note: '注意: <strong>A</strong> と <strong>B</strong> は、同じ単語ですがアクセントが異なる録音です。',
    step2Task: 'あなたのタスク：録音 <strong>X</strong> が、録音 <strong>A</strong> と <strong>B</strong> のどちらに近いと感じるか判断してください。',
    step2Title: "タスクの説明",
    step3Intro: "下の A、B、X の音声を聞いて、X が A と B のどちらに近いか選択してください。",
    step3Title: "練習",
    step4Title: "ヒント",
    step5Intro: "素晴らしい！それでは、本番の音声判別タスクを開始しましょう。",
    step5Title: "準備完了！",
    unableToInferDataset: "データセットを特定できませんでした。",
    unableToStart: "タスクを開始できません。",
    xSoundsLike: "X はどちらに近いですか？",
  },
};

function t(key, fallback) {
  const lang = STRINGS[state.dataset];
  return (lang && lang[key]) || fallback;
}

function applyTranslations() {
  const lang = STRINGS[state.dataset];
  if (!lang) return;
  if (lang.pageTitle) document.title = lang.pageTitle;
  document.querySelectorAll("[data-i18n]").forEach((el) => {
    const key = el.getAttribute("data-i18n");
    if (lang[key]) el.innerHTML = lang[key];
  });
}

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
  dataset: "",
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
  participantValue: document.getElementById("participantValue"),
  listLabel: document.getElementById("listLabel"),
  listValue: document.getElementById("listValue"),
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
    if (els.participantValue) els.participantValue.textContent = participantId;
  }

  if (list) {
    state.listId = list.split("/").pop() || list;
    if (els.listValue) els.listValue.textContent = state.listId;
  }
}

function inferDataset(listUrl) {
  const datasets = ["stress", "pitch_accent", "tone"];
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
  const ready = t("clipReady", "Ready");
  els.stateA.textContent = ready;
  els.stateB.textContent = ready;
  els.stateX.textContent = ready;
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
    setTaskHint(t("missingAudioSource", "Missing audio source. Please refresh and try again."));
    return false;
  }
  audioEl.src = url;
  audioEl.currentTime = 0;
  const playing = t("clipPlaying", "Playing {label}").replace("{label}", label);
  setClipState(clipEl, stateEl, playing);
  try {
    await audioEl.play();
  } catch (err) {
    clipEl.classList.remove("active");
    stateEl.textContent = t("clipBlocked", "Blocked");
    setTaskHint(t("playbackFailedReplay", "Audio playback failed. Please allow audio for this site and press Replay."));
    return false;
  }
  await new Promise((resolve) => {
    audioEl.onended = () => resolve();
  });
  clipEl.classList.remove("active");
  stateEl.textContent = t("clipPlayed", "Played");
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
  els.phaseLabel.textContent = t("inProgress", "In progress");
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
      if (doneTitle) doneTitle.textContent = t("bonusTitle", "Congratulations!");
      if (doneMessage) {
        doneMessage.innerHTML = t(
          "bonusMessage",
          "<strong>Great job!</strong> Your high accuracy earned you a bonus payment."
        );
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
    const completionLabel = t("completionCodeLabel", "Completion code");
    els.completionCode.textContent = `${completionLabel}: ${config.completionCode}`;
  } else {
    // Non-Prolific: no completion codes or redirects, just show download
    els.doneCompletionRow.classList.remove("hidden");
    els.completionCode.classList.add("hidden");
  }

  if (!result.ok && config.submitUrl) {
    if (result.error) {
      const errorPrefix = t("errorPrefix", "Error");
      els.doneError.textContent = `${errorPrefix}: ${result.error}`;
    }
    els.doneError.classList.remove("hidden");
    setTaskHint(t("autoSubmitFailed", "Auto-submit failed. Please download your responses."));
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
  setIntroHint(t("loadingList", "Loading list..."));
  state.loading = true;
  try {
    const res = await fetch(listUrl);
    if (!res.ok) {
      const message = t("failedToLoadListStatus", "Failed to load list ({status})").replace(
        "{status}",
        res.status
      );
      throw new Error(message);
    }
    const text = await res.text();
    const raw = parseCsv(text);
    state.orderedTrials = raw.map(normalizeTrial);
    enableStart(true);
    setIntroHint("");
    checkAndPromptResume();
  } catch (err) {
    els.errorText.textContent = err.message || t("failedToLoadList", "Failed to load list.");
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
      return {
        ok: false,
        error: t("missingAttentionFailUrl", "Missing attention_fail_url from server."),
      };
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
    els.headphonesHint.textContent = t("headphonesWarning", "Please put on wired headphones and find a quiet spot before continuing.");
  });

  els.toStep3.addEventListener("click", () => {
    showStep(3);
    els.demoPlay.disabled = false;
  });

  let demoFirstPlay = true;

  async function playDemoClip(audioEl, url, clipEl, stateEl, label) {
    clipEl.classList.add("active");
    const playing = t("clipPlaying", "Playing {label}").replace("{label}", label);
    stateEl.textContent = playing;
    audioEl.src = url;
    audioEl.currentTime = 0;
    try {
      await audioEl.play();
    } catch (err) {
      clipEl.classList.remove("active");
      stateEl.textContent = t("clipBlocked", "Blocked");
      els.demoHint.textContent = t(
        "playbackFailedTryAgain",
        "Audio playback failed. Please allow audio for this site and try again."
      );
      return false;
    }
    await new Promise((r) => (audioEl.onended = r));
    stateEl.textContent = t("clipPlayed", "Played");
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
    const ready = t("clipReady", "Ready");
    els.demoStateA.textContent = ready;
    els.demoStateB.textContent = ready;
    els.demoStateX.textContent = ready;

    await sleep(config.preDelayMs);

    // Play A
    const okA = await playDemoClip(els.audioA, urlA, els.demoClipA, els.demoStateA, "A");
    if (!okA) return false;

    await sleep(config.isiMs);

    // Play B
    const okB = await playDemoClip(els.audioB, urlB, els.demoClipB, els.demoStateB, "B");
    if (!okB) return false;

    await sleep(config.isiMs);

    // Play X
    const okX = await playDemoClip(els.audioX, urlX, els.demoClipX, els.demoStateX, "X");
    if (!okX) return false;
    return true;
  }

  els.demoPlay.addEventListener("click", async () => {
    els.demoPlay.disabled = true;
    els.demoChooseA.disabled = true;
    els.demoChooseB.disabled = true;
    els.demoChoiceLabel.style.visibility = "hidden";
    els.demoHint.textContent = "";

    // Demo files from demo_audio/{dataset}/ (X = A, so correct answer is A)
    const demo = DEMO_FILES[state.dataset];
    const urlA = `demo_audio/${state.dataset}/${demo.a}`;
    const urlB = `demo_audio/${state.dataset}/${demo.b}`;
    const urlX = urlA;

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
    els.demoPlay.innerHTML = t("replayDemo", "Replay A &rarr; B &rarr; X");
    els.demoHint.textContent = t("demoPrompt", "Now choose: does X sound more like A or B?");
  });

  const handleDemoChoice = (choice) => {
    // Demo correct answer is always A (since X = A)
    const correct = choice === "A";

    els.demoChooseA.disabled = true;
    els.demoChooseB.disabled = true;

    if (correct) {
      els.demoHint.textContent = t("demoCorrect", "Correct! You've got it.");
      setTimeout(() => {
        showStep(4);
      }, 800);
    } else {
      showFeedback(
        t("demoWrongTitle", "Not quite"),
        t("demoWrongMsg", "Listen carefully to A and B, then decide which one X sounds closer to. Let's try again."),
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
    throw new Error(
      t("missingListParam", "Missing required parameter: list (or dataset for auto-assignment)")
    );
  }
  state.listPath = listUrl;
  const inferredDataset = inferDataset(listUrl);
  if (!inferredDataset) {
    throw new Error(t("unableToInferDataset", "Unable to infer dataset from list path."));
  }
  state.dataset = inferredDataset;
  applyTranslations();
  config.audioBaseUrl = `audio/${inferredDataset}`;
  config.audioBaseUrlSyn = `audio/${inferredDataset}_syn`;
  state.listId = listUrl.split("/").pop() || listUrl;
  if (els.listValue) els.listValue.textContent = state.listId;
  loadTrialsFromUrl(listUrl);
}

window.addEventListener("load", () => {
  try {
    applyConfigFromParams();
    if (!config.audioBaseUrl) {
      throw new Error(t("missingAudioBase", "Missing required parameter: audioBase"));
    }
    attachHandlers();
    wireStart();
    wireIntroSteps();
    setListFromParams();

    if (isProlific) {
      const completionLabel = t("completionCodeLabel", "Completion code");
      els.completionCode.textContent = `${completionLabel}: ${config.completionCode}`;
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
    els.errorText.textContent = err.message || t("unableToStart", "Unable to start.");
    showCard(els.errorCard);
  }
});
