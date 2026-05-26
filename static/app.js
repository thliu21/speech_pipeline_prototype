const state = {
  devices: [],
  selectedDeviceId: null,
  running: false,
  recording: false,
  recordings: [],
  transcriptSegments: new Map(),
  pipelineBusy: false,
  recordingBusy: false,
};

const el = {
  statusLine: document.querySelector("#status-line"),
  runningPill: document.querySelector("#running-pill"),
  vadPill: document.querySelector("#vad-pill"),
  wsState: document.querySelector("#ws-state"),
  refreshDevices: document.querySelector("#refresh-devices"),
  startMic: document.querySelector("#start-mic"),
  stop: document.querySelector("#stop"),
  runWav: document.querySelector("#run-wav"),
  clearTranscript: document.querySelector("#clear-transcript"),
  startRecording: document.querySelector("#start-recording"),
  stopRecording: document.querySelector("#stop-recording"),
  refreshRecordings: document.querySelector("#refresh-recordings"),
  recordingPill: document.querySelector("#recording-pill"),
  recordingTitle: document.querySelector("#recording-title"),
  recordingReference: document.querySelector("#recording-reference"),
  recordingDetail: document.querySelector("#recording-detail"),
  recordingsList: document.querySelector("#recordings-list"),
  deviceSelect: document.querySelector("#device-select"),
  deviceDetail: document.querySelector("#device-detail"),
  wavPath: document.querySelector("#wav-path"),
  wavRealtime: document.querySelector("#wav-realtime"),
  currentSource: document.querySelector("#current-source"),
  inputDevice: document.querySelector("#input-device"),
  queueDepth: document.querySelector("#queue-depth"),
  lastError: document.querySelector("#last-error"),
  audioLevel: document.querySelector("#audio-level"),
  partial: document.querySelector("#partial"),
  transcript: document.querySelector("#transcript"),
  events: document.querySelector("#events"),
};

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`;
    try {
      const body = await response.json();
      message = body.detail || message;
    } catch (_) {}
    throw new Error(message);
  }
  return response.json();
}

async function refreshStatus() {
  const status = await api("/api/status");
  applyStatus(status);
}

async function refreshDevices() {
  try {
    const data = await api("/api/audio/devices");
    state.devices = data.devices || [];
    state.selectedDeviceId = data.selected_device_id;
    renderDevices();
    applyControlState();
  } catch (error) {
    setError(error.message);
  }
}

async function refreshRecordings() {
  try {
    const data = await api("/api/recordings");
    state.recording = Boolean(data.recording);
    state.recordings = data.recordings || [];
    renderRecordingStatus(data.active || null);
    renderRecordings();
    applyControlState();
  } catch (error) {
    setError(error.message);
  }
}

function renderDevices() {
  el.deviceSelect.innerHTML = "";
  if (!state.devices.length) {
    const option = document.createElement("option");
    option.textContent = "No input devices found";
    option.value = "";
    el.deviceSelect.append(option);
    el.deviceDetail.textContent = "Install PortAudio/sounddevice and connect a microphone.";
    return;
  }
  for (const device of state.devices) {
    const option = document.createElement("option");
    option.value = String(device.id);
    option.textContent = `${device.name}${device.is_default ? " (default)" : ""}`;
    el.deviceSelect.append(option);
  }
  const selected = state.selectedDeviceId ?? state.devices.find((d) => d.is_default)?.id ?? state.devices[0].id;
  el.deviceSelect.value = String(selected);
  updateDeviceDetail();
}

function updateDeviceDetail() {
  const id = selectedDeviceIdForRequest();
  const device = state.devices.find((item) => item.id === id);
  if (!device) {
    el.deviceDetail.textContent = "No microphone selected.";
    return;
  }
  el.deviceDetail.textContent =
    `#${device.id} | ${device.host_api} | ${device.max_input_channels} input channel(s) | ` +
    `${Math.round(device.default_samplerate)} Hz default`;
}

function selectedDeviceIdForRequest() {
  const value = el.deviceSelect.value;
  if (value === "") return null;
  const id = Number(value);
  return Number.isFinite(id) ? id : null;
}

async function selectDevice() {
  const id = selectedDeviceIdForRequest();
  if (id == null) return;
  try {
    clearError();
    await api("/api/audio/select_device", {
      method: "POST",
      body: JSON.stringify({ device_id: id }),
    });
    state.selectedDeviceId = id;
    updateDeviceDetail();
    addEvent("device", `Selected microphone #${id}`);
  } catch (error) {
    setError(error.message);
    await refreshDevices();
  }
}

async function startMic() {
  state.pipelineBusy = true;
  clearError();
  el.statusLine.textContent = "Starting microphone...";
  el.runningPill.textContent = "starting";
  el.runningPill.className = "pill warn";
  applyControlState();
  try {
    await api("/api/start", {
      method: "POST",
      body: JSON.stringify({ source: "mic", device_id: selectedDeviceIdForRequest() }),
    });
    addEvent("pipeline", "Started microphone");
    await refreshStatus();
  } catch (error) {
    el.statusLine.textContent = "Pipeline idle";
    el.runningPill.textContent = "idle";
    el.runningPill.className = "pill";
    setError(error.message);
  } finally {
    state.pipelineBusy = false;
    applyControlState();
  }
}

async function startRecording() {
  state.recordingBusy = true;
  clearError();
  el.recordingPill.textContent = "starting";
  el.recordingPill.className = "pill warn";
  el.recordingDetail.textContent = "Starting recording...";
  applyControlState();
  try {
    await api("/api/recordings/start", {
      method: "POST",
      body: JSON.stringify({
        device_id: selectedDeviceIdForRequest(),
        title: el.recordingTitle.value,
        reference_text: el.recordingReference.value,
      }),
    });
    addEvent("recording", "Started recording");
    await refreshRecordings();
  } catch (error) {
    el.recordingPill.textContent = "idle";
    el.recordingPill.className = "pill";
    el.recordingDetail.textContent = `Could not start recording: ${error.message}`;
    setError(error.message);
  } finally {
    state.recordingBusy = false;
    applyControlState();
  }
}

async function stopRecording() {
  state.recordingBusy = true;
  clearError();
  el.recordingDetail.textContent = "Stopping recording...";
  applyControlState();
  try {
    const data = await api("/api/recordings/stop", { method: "POST", body: "{}" });
    el.recordingTitle.value = "";
    addEvent("recording", `Saved ${data.recording?.title || data.recording?.id || "recording"}`);
    await refreshRecordings();
  } catch (error) {
    el.recordingDetail.textContent = `Could not stop recording: ${error.message}`;
    setError(error.message);
  } finally {
    state.recordingBusy = false;
    applyControlState();
  }
}

async function stopPipeline() {
  state.pipelineBusy = true;
  clearError();
  el.statusLine.textContent = "Stopping pipeline...";
  applyControlState();
  try {
    await api("/api/stop", { method: "POST", body: "{}" });
    addEvent("pipeline", "Stopped");
    await refreshStatus();
  } catch (error) {
    setError(error.message);
  } finally {
    state.pipelineBusy = false;
    applyControlState();
  }
}

async function runWav() {
  const path = el.wavPath.value.trim();
  if (!path) {
    setError("Enter a WAV path first.");
    return;
  }
  state.pipelineBusy = true;
  clearError();
  el.statusLine.textContent = "Starting WAV...";
  applyControlState();
  try {
    await api("/api/run_wav", {
      method: "POST",
      body: JSON.stringify({ path, realtime: el.wavRealtime.checked }),
    });
    addEvent("pipeline", "Started WAV");
    await refreshStatus();
  } catch (error) {
    setError(error.message);
  } finally {
    state.pipelineBusy = false;
    applyControlState();
  }
}

async function runRecording(recording) {
  state.pipelineBusy = true;
  clearError();
  el.statusLine.textContent = "Starting recording playback...";
  applyControlState();
  try {
    await api("/api/run_wav", {
      method: "POST",
      body: JSON.stringify({
        path: recording.path,
        realtime: el.wavRealtime.checked,
        reference_text: recording.reference_text || null,
      }),
    });
    addEvent("pipeline", `Started ${recording.title || recording.id}`);
    await refreshStatus();
  } catch (error) {
    setError(error.message);
  } finally {
    state.pipelineBusy = false;
    applyControlState();
  }
}

function connectSocket() {
  const protocol = window.location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${protocol}://${window.location.host}/ws/events`);
  ws.addEventListener("open", () => {
    el.wsState.textContent = "connected";
    el.wsState.className = "pill active";
  });
  ws.addEventListener("close", () => {
    el.wsState.textContent = "reconnecting";
    el.wsState.className = "pill warn";
    setTimeout(connectSocket, 1000);
  });
  ws.addEventListener("message", (message) => {
    const event = JSON.parse(message.data);
    handleEvent(event);
  });
}

function handleEvent(event) {
  const payload = event.payload || {};
  if (event.type === "status") applyStatus(payload);
  if (event.type === "metrics") updateMetric(payload);
  if (event.type === "pipeline_state") updatePipelineState(payload);
  if (event.type === "transcript") updateTranscript(payload);
  if (event.type === "segment") {
    if (payload.kind === "speech_end") el.partial.textContent = "";
    addEvent("segment", `${payload.kind} ${payload.start_ms ?? ""}-${payload.end_ms ?? ""}`);
  }
  if (event.type === "summary") addEvent("summary", `RTF ${payload.wall_rtf ?? "-"} | ${payload.audio_duration_sec}s audio`);
  if (event.type === "error") setError(payload.message || "Unknown error");
}

function applyStatus(status) {
  state.running = Boolean(status.running);
  el.statusLine.textContent = state.running ? "Pipeline running" : "Pipeline idle";
  el.runningPill.textContent = state.running ? "running" : "idle";
  el.runningPill.className = state.running ? "pill active" : "pill";
  el.currentSource.textContent = status.current_source || "-";
  el.inputDevice.textContent = status.current_input_device || "-";
  el.lastError.textContent = status.last_error || "-";
  applyControlState();
}

function applyControlState() {
  const busy = state.pipelineBusy || state.recordingBusy;
  el.deviceSelect.disabled = busy || state.running || state.recording;
  el.refreshDevices.disabled = busy || state.running || state.recording;
  el.startMic.disabled = busy || state.running || state.recording;
  el.runWav.disabled = busy || state.running || state.recording;
  el.stop.disabled = busy || !state.running;
  el.startRecording.disabled = busy || state.running || state.recording;
  el.stopRecording.disabled = busy || !state.recording;
  el.recordingTitle.disabled = busy || state.recording;
  el.recordingReference.disabled = busy || state.recording;
}

function updatePipelineState(payload) {
  el.vadPill.textContent = payload.vad || "silence";
  el.vadPill.className = payload.vad === "speech" ? "pill active" : "pill";
  el.queueDepth.textContent = String(payload.queue_depth ?? 0);
  el.inputDevice.textContent = payload.input_device || "-";
  const width = Math.min(100, Math.round((payload.audio_level || 0) * 180));
  el.audioLevel.style.width = `${width}%`;
}

function updateMetric(payload) {
  const card = document.querySelector(`.metric-card[data-stage="${payload.stage}"]`);
  if (!card) return;
  card.querySelector("strong").textContent = Number(payload.latest_ms || 0).toFixed(1);
  card.querySelector("small").textContent =
    `avg ${Number(payload.avg_ms || 0).toFixed(1)} / p95 ${Number(payload.p95_ms || 0).toFixed(1)}`;
}

function updateTranscript(payload) {
  if (payload.type === "partial") {
    el.partial.textContent = payload.text;
    return;
  }
  el.partial.textContent = "";
  const segmentId = payload.segment_id == null ? null : String(payload.segment_id);
  const existing = segmentId ? state.transcriptSegments.get(segmentId) : null;
  if (existing) {
    existing.textContent = payload.text;
    existing.dataset.revision = String(payload.revision ?? 1);
    return;
  }
  const line = document.createElement("p");
  line.textContent = payload.text;
  if (segmentId) {
    line.dataset.segmentId = segmentId;
    line.dataset.revision = String(payload.revision ?? 1);
    state.transcriptSegments.set(segmentId, line);
  }
  el.transcript.append(line);
}

function renderRecordingStatus(active) {
  el.recordingPill.textContent = state.recording ? "recording" : "idle";
  el.recordingPill.className = state.recording ? "pill active" : "pill";
  if (!active) {
    el.recordingDetail.textContent = "Recordings are saved on this Pi.";
    return;
  }
  el.recordingDetail.textContent =
    `${active.title || active.id} | ${formatDuration(active.duration_sec)} | ` +
    `device ${active.device_id ?? "-"}`;
}

function renderRecordings() {
  el.recordingsList.textContent = "";
  if (!state.recordings.length) {
    const empty = document.createElement("div");
    empty.className = "detail";
    empty.textContent = "No recordings yet.";
    el.recordingsList.append(empty);
    return;
  }
  for (const recording of state.recordings) {
    el.recordingsList.append(renderRecordingItem(recording));
  }
}

function renderRecordingItem(recording) {
  const item = document.createElement("article");
  item.className = "recording-item";

  const row = document.createElement("div");
  row.className = "recording-row";

  const info = document.createElement("div");
  const title = document.createElement("p");
  title.className = "recording-title";
  title.textContent = recording.title || recording.id;
  const meta = document.createElement("div");
  meta.className = "recording-meta";
  meta.textContent =
    `${recording.started_at || "-"} | ${formatDuration(recording.duration_sec)} | ` +
    `${formatBytes(recording.size_bytes)} | ${recording.status}`;
  info.append(title, meta);

  const tools = document.createElement("div");
  tools.className = "recording-tools";
  const useButton = button("Use WAV", () => {
    el.wavPath.value = recording.path;
    el.recordingReference.value = recording.reference_text || "";
  });
  const runButton = button("Run", () => runRecording(recording));
  const deleteButton = button("Delete", () => deleteRecording(recording.id));
  tools.append(useButton, runButton, deleteButton);
  row.append(info, tools);

  const audio = document.createElement("audio");
  audio.controls = true;
  audio.src = `/api/recordings/${encodeURIComponent(recording.id)}/audio`;

  const referenceRow = document.createElement("div");
  referenceRow.className = "recording-reference-row";
  const reference = document.createElement("textarea");
  reference.rows = 2;
  reference.value = recording.reference_text || "";
  reference.placeholder = "Reference transcript";
  const saveButton = button("Save Ref", () => saveRecordingReference(recording.id, reference.value));
  referenceRow.append(reference, saveButton);

  item.append(row, audio, referenceRow);
  return item;
}

function button(label, onClick) {
  const element = document.createElement("button");
  element.type = "button";
  element.textContent = label;
  element.addEventListener("click", onClick);
  return element;
}

async function saveRecordingReference(id, referenceText) {
  try {
    await api(`/api/recordings/${encodeURIComponent(id)}`, {
      method: "PATCH",
      body: JSON.stringify({ reference_text: referenceText }),
    });
    await refreshRecordings();
  } catch (error) {
    setError(error.message);
  }
}

async function deleteRecording(id) {
  try {
    await api(`/api/recordings/${encodeURIComponent(id)}`, { method: "DELETE" });
    await refreshRecordings();
  } catch (error) {
    setError(error.message);
  }
}

function formatDuration(seconds) {
  const value = Number(seconds || 0);
  if (value < 60) return `${value.toFixed(1)}s`;
  const minutes = Math.floor(value / 60);
  return `${minutes}m ${Math.round(value % 60)}s`;
}

function formatBytes(bytes) {
  const value = Number(bytes || 0);
  if (value < 1024) return `${value} B`;
  if (value < 1024 * 1024) return `${(value / 1024).toFixed(1)} KB`;
  return `${(value / (1024 * 1024)).toFixed(1)} MB`;
}

function addEvent(kind, message) {
  const item = document.createElement("li");
  const time = new Date().toLocaleTimeString();
  item.textContent = `${time} | ${kind} | ${message}`;
  el.events.prepend(item);
  while (el.events.children.length > 80) {
    el.events.lastElementChild.remove();
  }
}

function setError(message) {
  el.lastError.textContent = message;
  addEvent("error", message);
}

function clearError() {
  el.lastError.textContent = "-";
}

el.refreshDevices.addEventListener("click", refreshDevices);
el.refreshRecordings.addEventListener("click", refreshRecordings);
el.deviceSelect.addEventListener("change", selectDevice);
el.startMic.addEventListener("click", startMic);
el.stop.addEventListener("click", stopPipeline);
el.runWav.addEventListener("click", runWav);
el.startRecording.addEventListener("click", startRecording);
el.stopRecording.addEventListener("click", stopRecording);
el.clearTranscript.addEventListener("click", () => {
  el.partial.textContent = "";
  el.transcript.textContent = "";
  state.transcriptSegments.clear();
});

refreshStatus();
refreshDevices();
refreshRecordings();
connectSocket();

setInterval(() => {
  if (state.recording) refreshRecordings();
}, 1000);
