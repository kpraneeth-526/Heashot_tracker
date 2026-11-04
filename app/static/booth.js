setupCamera({
  endpoint: "/capture",
  describe: (body) => {
    if (!body.ok) return body.reason;
    if (body.match) return `Hi ${body.match.name}, is this you?`;
    return "We couldn't find you automatically. Please enter your email.";
  },
});
