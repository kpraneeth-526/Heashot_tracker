const nameInput = document.getElementById("name");
const emailInput = document.getElementById("email");
const idInput = document.getElementById("uga-id");
const saveBox = document.getElementById("save");

setupCamera({
  endpoint: "/enroll",
  headers: () => ({
    "X-Name": encodeURIComponent(nameInput.value.trim()),
    "X-Email": encodeURIComponent(emailInput.value.trim()),
    "X-Uga-Id": encodeURIComponent(idInput.value.trim()),
    "X-Save": saveBox.checked ? "1" : "0",
  }),
  check: () =>
    nameInput.value.trim() && emailInput.value.includes("@")
      ? null
      : "Please enter a name and a valid email first.",
  describe: (body) => {
    if (!body.ok) return body.reason;
    nameInput.value = "";
    emailInput.value = "";
    idInput.value = "";
    saveBox.checked = false; // each person decides for themselves
    const kept = body.saved ? " Saved for next time." : " For this session only.";
    return `Registered ${body.name}.${kept} ${body.gallery_size} people in the gallery.`;
  },
});
