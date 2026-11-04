const nameInput = document.getElementById("name");
const emailInput = document.getElementById("email");

setupCamera({
  endpoint: "/enroll",
  headers: () => ({
    "X-Name": encodeURIComponent(nameInput.value.trim()),
    "X-Email": encodeURIComponent(emailInput.value.trim()),
  }),
  check: () =>
    nameInput.value.trim() && emailInput.value.includes("@")
      ? null
      : "Please enter a name and a valid email first.",
  describe: (body) => {
    if (!body.ok) return body.reason;
    nameInput.value = "";
    emailInput.value = "";
    return `Registered ${body.name}. ${body.gallery_size} people in the gallery.`;
  },
});
