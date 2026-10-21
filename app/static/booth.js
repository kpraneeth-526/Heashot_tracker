const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const statusEl = document.getElementById("status");
const captureBtn = document.getElementById("capture");
const retakeBtn = document.getElementById("retake");

function setStatus(text) {
  statusEl.textContent = text;
}

async function startCamera() {
  try {
    video.srcObject = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1920 }, height: { ideal: 1080 }, facingMode: "user" },
      audio: false,
    });
    await video.play();
    captureBtn.disabled = false;
    setStatus("Look at the camera and press Take photo.");
  } catch (err) {
    setStatus(`Camera unavailable: ${err.message}`);
  }
}

function showLive() {
  canvas.hidden = true;
  video.hidden = false;
  retakeBtn.hidden = true;
  captureBtn.hidden = false;
  captureBtn.disabled = false;
  setStatus("Look at the camera and press Take photo.");
}

async function capture() {
  if (captureBtn.disabled || captureBtn.hidden) return;
  captureBtn.disabled = true;

  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  canvas.getContext("2d").drawImage(video, 0, 0);
  video.hidden = true;
  canvas.hidden = false;
  setStatus("Checking photo…");

  const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", 0.92));
  try {
    const res = await fetch("/capture", {
      method: "POST",
      headers: { "Content-Type": "image/jpeg" },
      body: blob,
    });
    const body = await res.json();
    if (!res.ok) throw new Error(body.detail || res.statusText);
    const n = body.faces;
    setStatus(n === 1 ? "Got it: one face found." : `Found ${n} faces — please retake with just one person.`);
  } catch (err) {
    setStatus(`Something went wrong: ${err.message}`);
  }
  captureBtn.hidden = true;
  retakeBtn.hidden = false;
}

captureBtn.addEventListener("click", capture);
retakeBtn.addEventListener("click", showLive);
document.addEventListener("keydown", (e) => {
  if (e.code !== "Space") return;
  e.preventDefault();
  if (!retakeBtn.hidden) showLive();
  else capture();
});

startCamera();
