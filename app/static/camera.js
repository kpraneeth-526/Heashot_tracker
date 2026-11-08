// Shared webcam capture for the booth (/) and registration (/enroll) pages.
// If the page has an #upload button and #file input, photos can also be
// chosen from the computer.
//
// setupCamera({
//   endpoint: "/capture",          // where the JPEG is POSTed
//   headers: () => ({}),           // extra request headers
//   check: () => null,             // return a message to block capture, or null
//   describe: (body) => "text",    // turn the JSON response into a status line
//   onResult: (body) => {},        // called after a successful response
//   onReset: () => {},             // called when going back to the live camera
// })
//
// Returns { reset } so the page can go back to the live camera itself.
function setupCamera({
  endpoint,
  headers = () => ({}),
  check = () => null,
  describe,
  onResult = () => {},
  onReset = () => {},
}) {
  const video = document.getElementById("video");
  const canvas = document.getElementById("canvas");
  const statusEl = document.getElementById("status");
  const captureBtn = document.getElementById("capture");
  const retakeBtn = document.getElementById("retake");
  const uploadBtn = document.getElementById("upload");
  const fileInput = document.getElementById("file");
  const prompt = uploadBtn
    ? "Look at the camera and press Take photo, or upload a photo."
    : "Look at the camera and press Take photo.";
  let cameraReady = false;

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
      cameraReady = true;
      captureBtn.disabled = false;
      setStatus(prompt);
    } catch (err) {
      setStatus(`Camera unavailable: ${err.message}` + (uploadBtn ? " You can still upload a photo." : ""));
    }
  }

  function showLive() {
    canvas.hidden = true;
    video.hidden = false;
    retakeBtn.hidden = true;
    captureBtn.hidden = false;
    captureBtn.disabled = !cameraReady;
    if (uploadBtn) uploadBtn.hidden = uploadBtn.disabled = false;
    setStatus(prompt);
    onReset();
  }

  // Draw a picture onto the canvas, shrunk to at most maxSide pixels.
  // Camera photos are displayed mirrored to match the live preview; uploads aren't.
  function drawToCanvas(source, width, height, maxSide = Infinity) {
    canvas.classList.toggle("mirrored", source === video);
    const scale = Math.min(1, maxSide / Math.max(width, height));
    canvas.width = Math.round(width * scale);
    canvas.height = Math.round(height * scale);
    canvas.getContext("2d").drawImage(source, 0, 0, canvas.width, canvas.height);
    video.hidden = true;
    canvas.hidden = false;
  }

  async function capture() {
    if (captureBtn.disabled || captureBtn.hidden) return;
    const problem = check();
    if (problem) {
      setStatus(problem);
      return;
    }
    drawToCanvas(video, video.videoWidth, video.videoHeight);
    await send();
  }

  // Uploaded files are re-encoded as JPEG in the browser, which also converts
  // iPhone HEIC photos (Safari can read them; the server can't) and shrinks
  // large phone photos. EXIF rotation is applied by the browser.
  async function upload(file) {
    const problem = check();
    if (problem) {
      setStatus(problem);
      return;
    }
    const url = URL.createObjectURL(file);
    try {
      const img = new Image();
      img.src = url;
      await img.decode();
      drawToCanvas(img, img.naturalWidth, img.naturalHeight, 1920);
    } catch {
      setStatus("Couldn't open that file. Please choose a JPG, PNG or HEIC photo.");
      return;
    } finally {
      URL.revokeObjectURL(url);
    }
    await send();
  }

  async function send() {
    captureBtn.disabled = true;
    if (uploadBtn) uploadBtn.disabled = true;
    setStatus("Checking photo…");

    const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", 0.92));
    try {
      const res = await fetch(endpoint, {
        method: "POST",
        headers: { "Content-Type": "image/jpeg", ...headers() },
        body: blob,
      });
      const body = await res.json();
      if (!res.ok) throw new Error(body.detail || res.statusText);
      setStatus(describe(body));
      onResult(body);
    } catch (err) {
      setStatus(`Something went wrong: ${err.message}`);
    }
    captureBtn.hidden = true;
    if (uploadBtn) uploadBtn.hidden = true;
    retakeBtn.hidden = false;
  }

  captureBtn.addEventListener("click", capture);
  retakeBtn.addEventListener("click", showLive);
  if (uploadBtn) {
    uploadBtn.disabled = false;
    uploadBtn.addEventListener("click", () => fileInput.click());
    fileInput.addEventListener("change", () => {
      const file = fileInput.files[0];
      fileInput.value = ""; // so choosing the same file again still triggers
      if (file) upload(file);
    });
  }
  document.addEventListener("keydown", (e) => {
    // Space is for the camera, not for typing or for a focused button.
    if (e.code !== "Space" || ["INPUT", "BUTTON"].includes(e.target.tagName)) return;
    e.preventDefault();
    if (!retakeBtn.hidden) showLive();
    else capture();
  });

  startCamera();
  return { reset: showLive };
}
