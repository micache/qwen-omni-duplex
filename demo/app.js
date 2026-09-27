const sample = window.DUPLEX_DEMO;
if (!sample) throw new Error("The recorded sample is missing.");

const $ = (id) => document.getElementById(id);
const audio = $("input-audio");
const events = sample.events;
const total = events.at(-1).emitTime + 0.7;
const bars = sample.waveform.map((level) => {
  const bar = document.createElement("span");
  bar.className = "wavebar";
  bar.style.height = `${Math.round(5 + level * 49)}px`;
  $("waveform").append(bar);
  return bar;
});
const frames = events.map((event) => {
  const frame = document.createElement("span");
  frame.className = `frame ${event.kind.toLowerCase()}`;
  frame.title = `Frame ${event.frame}: ${event.kind}${event.delta ? ` ${event.delta}` : ""}`;
  $("frame-track").append(frame);
  return frame;
});
const chunks = [...document.querySelectorAll(".chunk")];

let running = false;
let elapsed = 0;
let startedAt = 0;
let request = null;

function stamp(seconds) {
  const whole = Math.floor(seconds);
  return `${String(Math.floor(whole / 60)).padStart(2, "0")}:${String(whole % 60).padStart(2, "0")}`;
}

function render(time) {
  const heard = Math.min(time, sample.userSeconds);
  const heardFraction = heard / sample.userSeconds;
  const count = events.findIndex((event) => event.emitTime > time);
  const emitted = count === -1 ? events.length : count;
  const latest = emitted ? events[emitted - 1] : null;
  const text = events.slice(0, emitted).filter((event) => event.kind === "TEXT").map((event) => event.delta).join("");
  const hasStart = events.slice(0, emitted).some((event) => event.kind === "START");
  const hasStop = events.slice(0, emitted).some((event) => event.kind === "STOP");

  bars.forEach((bar, index) => bar.classList.toggle("heard", index / bars.length < heardFraction && time > 0));
  frames.forEach((frame, index) => {
    frame.classList.toggle("done", index < emitted);
    frame.classList.toggle("focus", index === emitted - 1);
  });
  chunks.forEach((chunk, index) => {
    const end = Math.min((index + 1) * sample.chunkSeconds, sample.trainingSeconds);
    chunk.classList.toggle("active", latest?.chunk === index && time < total - 0.7);
    chunk.classList.toggle("complete", time >= end);
  });

  $("audio-clock").textContent = `${heard.toFixed(2)} / ${sample.userSeconds.toFixed(2)} s`;
  $("replay-clock").textContent = `${stamp(time)} / ${stamp(total)}`;
  $("decision-count").textContent = `${String(emitted).padStart(3, "0")} / ${events.length}`;
  $("answer").innerHTML = text ? "" : '<span class="placeholder">The response appears here, token by token.</span>';
  if (text) $("answer").textContent = text;
  $("cursor").classList.toggle("visible", hasStart && !hasStop);

  const kind = latest?.kind || "IDLE";
  $("current-event").textContent = kind;
  $("current-event").className = `event-pill ${kind.toLowerCase()}`;
  $("event-explainer").textContent = kind === "IDLE" ? "Listening without speaking" : kind === "START" ? "The model begins a response" : kind === "STOP" ? "Response complete" : "Text arrives one token at a time";
  $("input-state").textContent = time < sample.userSeconds ? "Spoken instruction" : time < sample.trainingSeconds ? "Assistant silence" : "Audio received";
  $("model-state").textContent = hasStop ? "Response complete" : hasStart ? "Writing response" : emitted ? "Processing audio frames" : "Waiting for audio";
}

function frame() {
  elapsed = Math.min(total, (performance.now() - startedAt) / 1000);
  render(elapsed);
  if (elapsed >= total) {
    running = false;
    document.body.classList.remove("is-playing");
    $("play-icon").textContent = "↺";
    $("play-label").textContent = "Replay sample";
    return;
  }
  request = requestAnimationFrame(frame);
}

function start() {
  if (elapsed >= total) elapsed = 0;
  running = true;
  startedAt = performance.now() - elapsed * 1000;
  if (elapsed < sample.userSeconds) {
    audio.currentTime = elapsed;
    audio.play().catch(() => {});
  }
  document.body.classList.add("is-playing");
  $("play-icon").textContent = "Ⅱ";
  $("play-label").textContent = "Pause replay";
  request = requestAnimationFrame(frame);
}

function pause() {
  running = false;
  if (request) cancelAnimationFrame(request);
  audio.pause();
  document.body.classList.remove("is-playing");
  $("play-icon").textContent = "▶";
  $("play-label").textContent = "Continue replay";
}

$("play-button").addEventListener("click", () => running ? pause() : start());
$("restart-button").addEventListener("click", () => {
  pause();
  elapsed = 0;
  audio.currentTime = 0;
  render(0);
  $("play-label").textContent = "Play replay";
});
render(0);
