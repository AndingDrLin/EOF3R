# AutoLab: 7×24 Automated Experiment Pipeline

## Quick Start

```bash
# Start the pipeline (runs forever, one cycle per hour)
nohup bash eof3r/scripts/autolab/orchestrator.sh > /dev/null 2>&1 &

# Or run N cycles interactively
# In Claude Code:
workflow(".claude/workflows/autolab_supervisor.py", args={'cycles': 5})
```

## Control

```bash
# Pause (finishes current cycle, then stops)
touch outputs/autolab/PAUSE

# Resume
rm outputs/autolab/PAUSE

# Stop gracefully
touch outputs/autolab/STOP

# Check status
cat outputs/autolab/state.json
cat outputs/autolab/autolab_log.md
```

## Architecture

```
orchestrator.sh          # Main loop, runs cycles hourly
  └─ claude -p           # Each cycle runs as a Claude Code session
     ├─ OBSERVE           # Read state, check GPU/disk
     ├─ ORIENT            # Analyze previous results
     ├─ DECIDE            # Pick next experiment from queue
     ├─ ACT               # Implement + run experiment
     └─ DOCUMENT          # Update logs, git commit
```

## Files

```
outputs/autolab/
├── state.json              # Pipeline state (cycle count, results)
├── experiment_queue.yaml   # Pending experiments
├── autolab_log.md          # Cycle-by-cycle log
├── ALERT.md                # Human notification (if issues)
├── orchestrator.log        # Orchestrator stdout
├── orchestrator.pid        # PID file
├── PAUSE                   # Touch to pause
├── STOP                    # Touch to stop
└── results/                # Per-experiment outputs
    ├── baseline_v1/
    ├── loss_ablation_no_chamfer/
    └── ...
```

## Safety Rules

| Rule | Limit | Action |
|------|-------|--------|
| GPU memory | < 90% | Halt + alert |
| Disk usage | < 85% | Halt + alert |
| Cycle time | < 2 hours | Kill experiment |
| Consecutive failures | < 3 | Halt + alert |
| NaN loss | — | Kill immediately |
| New packages | — | Forbidden |
| baselines/ edits | — | Forbidden |

## Adding Experiments

Edit `outputs/autolab/experiment_queue.yaml`:

```yaml
- name: my_experiment
  description: What this tests
  config:
    batch_size: 4
    total_steps: 30000
    # Override any training parameter
  priority: 2
  status: pending
  depends_on: [baseline_v1]
  rationale: "Why this experiment matters"
```

Then touch `outputs/autolab/PAUSE` and `rm` it to pick up changes.
