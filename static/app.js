const fileInput = document.getElementById("fileInput");
const runBtn = document.getElementById("runBtn");
const resetBtn = document.getElementById("resetBtn");
const statusEl = document.getElementById("status");
const originalImg = document.getElementById("originalImg");
const overlayImg = document.getElementById("overlayImg");
const maskImg = document.getElementById("maskImg");

let selectedFile = null;
let previewUrl = null;

function setStatus(message, isError = false) {
  statusEl.textContent = message;
  statusEl.classList.toggle("error", isError);
}

function clearImages() {
  originalImg.removeAttribute("src");
  overlayImg.removeAttribute("src");
  maskImg.removeAttribute("src");
}

fileInput.addEventListener("change", (event) => {
  const file = event.target.files && event.target.files[0];
  if (!file) {
    selectedFile = null;
    runBtn.disabled = true;
    setStatus("No image selected.");
    clearImages();
    return;
  }

  selectedFile = file;
  runBtn.disabled = false;
  setStatus("Ready to run segmentation.");

  if (previewUrl) {
    URL.revokeObjectURL(previewUrl);
  }
  previewUrl = URL.createObjectURL(file);
  originalImg.src = previewUrl;
  overlayImg.removeAttribute("src");
  maskImg.removeAttribute("src");
});

runBtn.addEventListener("click", async () => {
  if (!selectedFile) {
    return;
  }

  runBtn.disabled = true;
  setStatus("Running segmentation...");

  const data = new FormData();
  data.append("file", selectedFile);

  try {
    const response = await fetch("/api/segment", {
      method: "POST",
      body: data,
    });


    const contentType = response.headers.get("content-type") || "";
    const payload = contentType.includes("application/json")
       ? await response.json()
       : { error: await response.text() };
    if (!response.ok) {
      setStatus(payload.error || "Segmentation failed.", true);
      runBtn.disabled = false;
      return;
    }

    originalImg.src = payload.original;
    overlayImg.src = payload.overlay;
    maskImg.src = payload.mask;
    setStatus("Segmentation complete.");
  } catch (error) {
    setStatus(error.message || "Network error while running segmentation.", true);
  } finally {
    runBtn.disabled = false;
  }
});

resetBtn.addEventListener("click", () => {
  fileInput.value = "";
  selectedFile = null;
  runBtn.disabled = true;
  setStatus("No image selected.");
  clearImages();

  if (previewUrl) {
    URL.revokeObjectURL(previewUrl);
    previewUrl = null;
  }
});
