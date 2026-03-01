# Ethical RL Healthcare Simulation Project

This project implements your paper's proposed approach as a complete working reinforcement learning simulation:

- affordability-aware environment dynamics,
- Relative Financial Burden (RFB) tracking,
- Gini-based fairness penalty inside policy learning,
- realtime streaming dashboard with episode-by-episode updates.

## Included Algorithms

- Clinical Q-Learning (conventional RL objective)
- Cost-Sensitive Q-Learning
- Multi-Objective Q-Learning
- Primal-Dual Constrained Q-Learning (safe RL inspired)
- Proposed Fairness-Constrained Q-Learning (RFB + Gini penalty)

## UI Features

- live simulation stream using Server-Sent Events (SSE)
- automatic pre-simulation auto-tuning for fairness-constrained model parameters
- dark hacker terminal aesthetic
- code-style typography (Consolas/Cascadia/JetBrains Mono fallback)
- expanded analytics with multiple charts:
  - reward trend
  - gini trend
  - adherence trend
  - clinical trend
  - final reward comparison bar chart
  - cost vs equity scatter
  - performance radar
  - RFB and Gini grouped bars

## Project Structure

- `app.py` Flask app + API
- `rl/environment.py` synthetic healthcare environment with cost-induced adherence
- `rl/algorithms.py` RL training logic, epsilon-greedy policy, replay updates
- `rl/metrics.py` inequality metrics (Gini)
- `rl/simulation.py` multi-algorithm comparison orchestration
- `templates/index.html` frontend dashboard
- `static/styles.css` professional UI styling
- `static/app.js` charting + API integration

### API Endpoints

- `POST /api/simulate` full comparison response after completion
- `GET /api/simulate_stream?payload=<json>` realtime SSE stream

The stream can emit additional tuning events before training begins:

- `autotune_start`
- `autotune_result`

## Run

1. Create and activate a Python environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Start the app:

```bash
python app.py
```

4. Open:

```text
http://127.0.0.1:5000
```

## Notes on the Proposed Objective

The proposed mode operationalizes:

- affordability-aware transitions where non-adherence rises with financial burden,
- RFB per patient: cumulative cost / economic capacity,
- fairness-constrained reward adjustment with `lambda * Gini(RFB)`.

This keeps fairness inside optimization rather than as a post-hoc metric.
