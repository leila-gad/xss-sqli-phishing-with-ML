const API_URL = "http://127.0.0.1:8000";

async function analyze() {
  const payload = document.getElementById("payload").value.trim();
  const resultDiv = document.getElementById("result");
  const loadingDiv = document.getElementById("loading");

  if (!payload) {
    alert("Enter a payload first");
    return;
  }

  resultDiv.classList.add("hidden");
  loadingDiv.classList.remove("hidden");

  try {
    const response = await fetch(`${API_URL}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ payload })
    });

    const data = await response.json();
    console.log("API response:", data); 
    let cls = "attack-benign";
    if (data.prediction === "XSS") cls = "attack-xss";
    if (data.prediction === "SQL Injection") cls = "attack-sqli";
    if (data.prediction === "Phishing") cls = "attack-phishing";

    resultDiv.className = `result ${cls}`;
    resultDiv.innerHTML = `
      <p><strong>Prediction:</strong> ${data.prediction || "Unknown"}</p>
      <p><strong>Confidence:</strong> ${(data.confidence * 100).toFixed(2)}%</p>
      <p><strong>Severity:</strong> ${data.severity}</p>
    `;

  } catch (err) {
    console.error("Analyze error:", err);
    resultDiv.className = "result error";
    resultDiv.innerText = "API error. Check console for details.";
  }

  loadingDiv.classList.add("hidden");
  resultDiv.classList.remove("hidden");
  loadLogs(); 
}

async function loadLogs() {
  const table = document.getElementById("logsTable");
  table.innerHTML = "";

  try {
    const response = await fetch(`${API_URL}/logs`);
    const data = await response.json();
    const logs = data.logs || [];

    if (logs.length === 0) {
      table.innerHTML = "<tr><td colspan='4'>No logs yet</td></tr>";
      return;
    }

    logs.forEach(log => {
      const row = document.createElement("tr");
      row.innerHTML = `
        <td>${log.timestamp || "-"}</td>
        <td>${log.payload || "-"}</td>
        <td>${log.prediction || "-"}</td>
        <td>${log.confidence ? (log.confidence * 100).toFixed(2) : "-"}%</td>
      `;
      table.appendChild(row);
    });

  } catch (err) {
    console.error("Load logs error:", err);
    table.innerHTML = "<tr><td colspan='4'>Unable to load logs</td></tr>";
  }
}

document.addEventListener("DOMContentLoaded", () => {
  loadLogs();
});
