
import { useState, useEffect } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { ShieldAlert, Activity } from "lucide-react";

export default function IDSApp() {
  const [payload, setPayload] = useState("");
  const [result, setResult] = useState(null);
  const [logs, setLogs] = useState([]);

  const API_BASE = "http://127.0.0.1:8000";

  // Send payload to backend for prediction
  const predict = async () => {
    try {
      const res = await fetch(`${API_BASE}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ payload }),
      });
      const data = await res.json();
      setResult(data);
      loadLogs(); // refresh logs after prediction
    } catch (err) {
      console.error("Prediction error:", err);
    }
  };

  // Load recent logs
  const loadLogs = async () => {
    try {
      const res = await fetch(`${API_BASE}/logs?limit=10`);
      const data = await res.json();
      setLogs(data.logs || []);
    } catch (err) {
      console.error("Logs error:", err);
    }
  };

  useEffect(() => {
    loadLogs();
  }, []);

  return (
    <div className="min-h-screen p-8 bg-gray-50">
      <h1 className="text-3xl font-bold mb-6 flex items-center gap-2">
        <ShieldAlert /> Intrusion Detection Dashboard
      </h1>

      {/* Prediction Card */}
      <Card className="mb-6">
        <CardContent className="p-6">
          <h2 className="text-xl font-semibold mb-4">Payload Analysis</h2>
          <Input
            placeholder="Enter payload (URL, script, request...)"
            value={payload}
            onChange={(e) => setPayload(e.target.value)}
            className="mb-4"
          />
          <Button onClick={predict}>Analyze</Button>

          {result && (
            <div className="mt-4">
              <p>
                <strong>Prediction:</strong> {result.prediction}
              </p>
              <p>
                <strong>Confidence:</strong>{" "}
                {(result.confidence * 100).toFixed(2)}%
              </p>
              <p>
                <strong>Severity:</strong> {result.severity}
              </p>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Logs Card */}
      <Card>
        <CardContent className="p-6">
          <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
            <Activity /> Recent Logs
          </h2>
          <ul className="space-y-2">
            {logs.map((log, i) => (
              <li key={i} className="border rounded p-3 bg-white">
                <p>
                  <strong>{log.attack_type}</strong> | {log.severity}
                </p>
                <p className="text-sm text-gray-600">{log.payload}</p>
                <p className="text-xs text-gray-400">{log.timestamp}</p>
              </li>
            ))}
          </ul>
        </CardContent>
      </Card>
    </div>
  );
}
