from __future__ import annotations

import json

from flask import Flask, jsonify, render_template, request

from rl.simulation import (
    ALGORITHMS,
    explain_counterfactual,
    run_ablation_studio,
    run_comparison,
    run_comparison_stream,
    run_scenario_lab,
)

app = Flask(__name__, template_folder="templates", static_folder="static")


@app.get("/")
def index():
    return render_template("index.html", algorithms=ALGORITHMS)


@app.get("/api/algorithms")
def algorithms():
    return jsonify({"algorithms": ALGORITHMS})


@app.post("/api/simulate")
def simulate():
    payload = request.get_json(silent=True) or {}
    result = run_comparison(payload)
    return jsonify(result)


@app.post("/api/ablation")
def ablation():
    payload = request.get_json(silent=True) or {}
    result = run_ablation_studio(payload)
    return jsonify(result)


@app.post("/api/scenario_lab")
def scenario_lab():
    payload = request.get_json(silent=True) or {}
    result = run_scenario_lab(payload)
    return jsonify(result)


@app.post("/api/counterfactual")
def counterfactual():
    payload = request.get_json(silent=True) or {}
    result = explain_counterfactual(payload)
    return jsonify(result)


@app.get("/api/simulate_stream")
def simulate_stream():
    payload_raw = request.args.get("payload", "{}")
    try:
        payload = json.loads(payload_raw)
    except json.JSONDecodeError:
        payload = {}

    def event_stream():
        try:
            for event in run_comparison_stream(payload):
                event_type = event.get("type", "message")
                yield f"event: {event_type}\n"
                yield f"data: {json.dumps(event)}\n\n"
        except GeneratorExit:
            return
        except Exception as exc:
            event = {
                "type": "stream_error",
                "message": str(exc),
            }
            yield "event: stream_error\n"
            yield f"data: {json.dumps(event)}\n\n"

    return app.response_class(
        event_stream(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


if __name__ == "__main__":
    app.run(debug=True)
