// Shared camera capture for the booth (/) and registration (/enroll) pages.
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
// If the booth was started with a DSLR (DSLR=1), the page shows the DSLR's
// live view instead of the webcam, and Take photo fires the DSLR. Photos
// taken with the DSLR's own shutter button show up the same way. The
// full-size DSLR photo is sent to `endpoint` exactly like a webcam photo.
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
  const live = document.getElementById("live");
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
  let usingDslr = false;
  let livePaused = false; // DSLR busy taking a photo: hold the last frame

  function setStatus(text) {
    statusEl.textContent = text;
  }

  async function startCamera() {
    try {
      const res = await fetch("/camera");
      usingDslr = res.ok && (await res.json()).dslr;
    } catch {
      usingDslr = false;
    }
    if (usingDslr) return startLiveView();
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

  // DSLR live view: fetch the next frame as soon as the last one is shown
  // (about 6 a second on a Canon T3i), while the live view is on screen.
  function startLiveView() {
    video.hidden = true;
    live.hidden = false;
    setStatus("Connecting to the camera…");
    liveLoop();
  }

  async function liveLoop() {
    let wait = 0;
    if (!live.hidden && !livePaused) {
      try {
        const res = await fetch("/camera/frame");
        if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
        // Decode the frame fully first, then draw it in one go: the picture
        // goes straight from one frame to the next, never blank in between.
        const frame = await createImageBitmap(await res.blob());
        if (livePaused) {
          frame.close();
          throw null; // a photo is being taken: keep the frozen frame
        }
        if (live.width !== frame.width || live.height !== frame.height) {
          live.width = frame.width;
          live.height = frame.height;
        }
        live.getContext("2d").drawImage(frame, 0, 0);
        frame.close();
        if (res.headers.get("X-Shutter") === "1") shutterPressed();
        if (!cameraReady) {
          cameraReady = true;
          if (retakeBtn.hidden) {
            captureBtn.disabled = false;
            setStatus(prompt);
          }
        }
      } catch (err) {
        if (err === null) return setTimeout(liveLoop, 200);
        // Unplugged or asleep: say so, and keep trying.
        cameraReady = false;
        captureBtn.disabled = true;
        setStatus(`Camera: ${err.message}`);
        wait = 2000;
      }
    } else {
      wait = 200; // a photo is on screen; check again shortly
    }
    setTimeout(liveLoop, wait);
  }

  function showLive() {
    livePaused = false;
    canvas.hidden = true;
    video.hidden = usingDslr;
    live.hidden = !usingDslr;
    retakeBtn.hidden = true;
    captureBtn.hidden = false;
    captureBtn.disabled = !cameraReady;
    if (uploadBtn) uploadBtn.hidden = uploadBtn.disabled = false;
    setStatus(cameraReady || !usingDslr ? prompt : "Connecting to the camera…");
    onReset();
  }

  // Draw a picture onto the canvas, shrunk to at most maxSide pixels.
  // Camera photos are displayed mirrored to match the live preview; uploads aren't.
  function drawToCanvas(source, width, height, maxSide = Infinity, mirrored = source === video) {
    canvas.classList.toggle("mirrored", mirrored);
    const scale = Math.min(1, maxSide / Math.max(width, height));
    canvas.width = Math.round(width * scale);
    canvas.height = Math.round(height * scale);
    canvas.getContext("2d").drawImage(source, 0, 0, canvas.width, canvas.height);
    video.hidden = true;
    live.hidden = true;
    canvas.hidden = false;
  }

  async function capture() {
    if (captureBtn.disabled || captureBtn.hidden) return;
    const problem = check();
    if (problem) {
      setStatus(problem);
      return;
    }
    if (usingDslr) return captureDslr();
    drawToCanvas(video, video.videoWidth, video.videoHeight);
    await send();
  }

  // Someone pressed the DSLR's own shutter button.
  function shutterPressed() {
    if (!retakeBtn.hidden || livePaused) return;
    const problem = check(); // e.g. the registration form isn't filled in yet
    if (problem) {
      setStatus(problem); // the photo waits until it is
      return;
    }
    captureDslr("/camera/shutter");
  }

  // Fire the DSLR (or collect a shutter-button photo), show the photo, and
  // send the full-size original on.
  async function captureDslr(source = "/camera/take") {
    captureBtn.disabled = true;
    if (uploadBtn) uploadBtn.disabled = true;
    livePaused = true; // freeze the live view while the camera is busy
    setStatus("Taking photo… hold still");
    let photo;
    try {
      const res = await fetch(source, { method: "POST" });
      if (res.status === 204) throw new Error(prompt); // shutter photo already taken by another tab
      if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
      photo = await res.blob();
      const url = URL.createObjectURL(photo);
      try {
        const img = new Image();
        img.src = url;
        await img.decode();
        drawToCanvas(img, img.naturalWidth, img.naturalHeight, 1920, true);
      } finally {
        URL.revokeObjectURL(url);
      }
    } catch (err) {
      // No photo: go straight back to the live view and say what to fix.
      livePaused = false;
      captureBtn.disabled = false;
      if (uploadBtn) uploadBtn.disabled = false;
      setStatus(err.message);
      return;
    }
    await send(photo);
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
    try {
      await drawFile(file);
    } catch {
      // Safari can't open some iPhone HEIC photos; the booth server can.
      setStatus("Opening photo…");
      try {
        const res = await fetch("/convert", { method: "POST", body: file });
        if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
        await drawFile(await res.blob());
      } catch (err) {
        setStatus(err.message);
        return;
      }
    }
    await send();
  }

  async function drawFile(file) {
    const url = URL.createObjectURL(file);
    try {
      const img = new Image();
      img.src = url;
      await img.decode();
      drawToCanvas(img, img.naturalWidth, img.naturalHeight, 1920, false);
    } finally {
      URL.revokeObjectURL(url);
    }
  }

  // Send `photo` (the DSLR original), or else what's on the canvas.
  async function send(photo = null) {
    captureBtn.disabled = true;
    if (uploadBtn) uploadBtn.disabled = true;
    setStatus("Checking photo…");

    const blob = photo || (await new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", 0.92)));
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
