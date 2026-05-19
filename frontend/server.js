const express = require("express");
const path    = require("path");

// Load .env
try { require("fs").readFileSync(".env").toString().split("\n").forEach(line => {
  const [k,v] = line.split("="); if(k&&v) process.env[k.trim()] = v.trim();
}); } catch {}

const app     = express();
const PORT    = process.env.PORT    || 3000;
const API_URL = process.env.API_URL || "http://localhost:5000";

app.use(express.static(path.join(__dirname, "public")));

app.get("/", (req, res) => {
  res.sendFile(path.join(__dirname, "views", "index.html"));
});

// Proxy /api/* → Flask backend
app.get("/api/*", async (req, res) => {
  try {
    const fetch = (await import("node-fetch")).default;
    const qs    = req.url.split("?")[1] || "";
    const url   = `${API_URL}${req.path}${qs ? "?" + qs : ""}`;
    console.log(`→ ${url}`);
    const resp  = await fetch(url, { signal: AbortSignal.timeout(10000) });
    const data  = await resp.json();
    res.json(data);
  } catch (err) {
    res.status(503).json({
      error:   "Backend offline",
      message: "Start Flask: cd backend && python app.py",
      api_url: API_URL,
    });
  }
});

app.listen(PORT, () => {
  console.log(`\n🌫️  Pearls AQI → http://localhost:${PORT}`);
  console.log(`   Backend  → ${API_URL}\n`);
});
