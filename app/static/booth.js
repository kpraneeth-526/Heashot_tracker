// Booth flow after a photo:
//   match     -> "Hi X, is this you?"  [Yes, that's me] [Not me]
//   no match  -> find yourself by UGA ID or name -> pick your name
//   not found -> type your email
// The server hands back a ticket for each photo; emails and IDs stay there.

const sections = {
  confirm: document.getElementById("confirm"),
  lookup: document.getElementById("lookup"),
  manual: document.getElementById("manual"),
};
const statusEl = document.getElementById("status");
const queryInput = document.getElementById("query");
const resultsEl = document.getElementById("results");
const emailInput = document.getElementById("email");
const retakeBtn = document.getElementById("retake");

let ticket = null;
let searchTimer = null;

function show(name) {
  for (const [key, el] of Object.entries(sections)) el.hidden = key !== name;
}

async function post(path, body) {
  const res = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ticket, ...body }),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || res.statusText);
  return data;
}

function startLookup(message) {
  statusEl.textContent = message;
  queryInput.value = "";
  resultsEl.replaceChildren();
  show("lookup");
  queryInput.focus();
}

async function search() {
  const query = queryInput.value.trim();
  if (!query) {
    resultsEl.replaceChildren();
    return;
  }
  try {
    const { results } = await post("/lookup", { query });
    if (query !== queryInput.value.trim()) return; // a newer search is on its way
    resultsEl.replaceChildren(
      ...results.map(({ choice, name }) => {
        const btn = document.createElement("button");
        btn.textContent = name;
        btn.addEventListener("click", () => finish({ choice }));
        return btn;
      }),
    );
    if (!results.length) {
      const p = document.createElement("p");
      p.textContent = "No one found. Check the spelling, or enter your email below.";
      resultsEl.append(p);
    }
  } catch (err) {
    statusEl.textContent = err.message;
  }
}

async function finish(choice) {
  try {
    const { name } = await post("/confirm", choice);
    statusEl.textContent = name ? `Thanks, ${name}! You're all set.` : "Thanks! You're all set.";
    show(null);
    ticket = null;
    retakeBtn.textContent = "Next person";
  } catch (err) {
    statusEl.textContent = err.message;
  }
}

setupCamera({
  endpoint: "/capture",
  describe: (body) => {
    if (!body.ok) return body.reason;
    if (body.match) return `Hi ${body.match.name}, is this you?`;
    return "We couldn't find you automatically. Type your UGA ID or name.";
  },
  onResult: (body) => {
    if (!body.ok) return;
    ticket = body.ticket;
    if (body.match) show("confirm");
    else startLookup(statusEl.textContent);
  },
  onReset: () => {
    ticket = null;
    show(null);
    retakeBtn.textContent = "Retake";
  },
});

document.getElementById("yes").addEventListener("click", () => finish({ match: true }));
document.getElementById("not-me").addEventListener("click", () =>
  startLookup("Sorry about that! Type your UGA ID or name."),
);
document.getElementById("no-roster").addEventListener("click", () => {
  statusEl.textContent = "Enter your email and we'll send your headshot there.";
  emailInput.value = "";
  show("manual");
  emailInput.focus();
});
document.getElementById("manual-form").addEventListener("submit", (e) => {
  e.preventDefault();
  finish({ email: emailInput.value.trim() });
});
queryInput.addEventListener("input", () => {
  clearTimeout(searchTimer);
  searchTimer = setTimeout(search, 250);
});
