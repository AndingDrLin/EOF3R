#!/bin/bash
# ============================================================
# AutoLab Orchestrator — 7×24 Continuous Experiment Pipeline
# ============================================================
#
# This script runs AutoLab cycles continuously, one per hour.
# Each cycle: OBSERVE → ORIENT → DECIDE → ACT → DOCUMENT
#
# Usage:
#   bash eof3r/scripts/autolab/orchestrator.sh
#
# To pause:
#   touch outputs/autolab/PAUSE
#
# To resume:
#   rm outputs/autolab/PAUSE
#
# To stop gracefully:
#   touch outputs/autolab/STOP
#
# ============================================================

set -euo pipefail

PROJECT_ROOT="/home/ubuntu/lyj/Project/EOF3R"
OUTPUT_DIR="$PROJECT_ROOT/outputs/autolab"
PAUSE_FILE="$OUTPUT_DIR/PAUSE"
STOP_FILE="$OUTPUT_DIR/STOP"
LOG_FILE="$OUTPUT_DIR/orchestrator.log"
PID_FILE="$OUTPUT_DIR/orchestrator.pid"

# Conda setup
source /home/ubuntu/lyj/anaconda3/etc/profile.d/conda.sh

# Write PID for external management
echo $$ > "$PID_FILE"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

check_safety() {
    # GPU check
    local gpu_pct
    gpu_pct=$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null | awk -F',' '{printf "%.0f", $1/$2*100}' || echo "0")
    if [ "$gpu_pct" -gt 90 ]; then
        log "⚠️ GPU memory at ${gpu_pct}% — HALTING"
        echo "GPU memory exceeded 90%" > "$OUTPUT_DIR/ALERT.md"
        return 1
    fi

    # Disk check
    local disk_pct
    disk_pct=$(df "$PROJECT_ROOT" | tail -1 | awk '{print $5}' | tr -d '%')
    if [ "$disk_pct" -gt 85 ]; then
        log "⚠️ Disk usage at ${disk_pct}% — HALTING"
        echo "Disk usage exceeded 85%" > "$OUTPUT_DIR/ALERT.md"
        return 1
    fi

    return 0
}

run_cycle() {
    local cycle_num=$1
    log "━━━ Cycle $cycle_num START ━━━"

    # Check for pause/stop signals
    if [ -f "$PAUSE_FILE" ]; then
        log "⏸ PAUSED — remove $PAUSE_FILE to resume"
        return 0
    fi
    if [ -f "$STOP_FILE" ]; then
        log "🛑 STOP — removing stop file"
        rm -f "$STOP_FILE"
        return 1
    fi

    # Safety check
    if ! check_safety; then
        return 1
    fi

    # Run the cycle via Claude Code
    # Activate eof3r env for the orchestrator
    conda activate eof3r

    log "Running cycle $cycle_num via Claude Code..."

    # Use claude code to run the cycle
    # The agent will read state, decide what to do, and execute
    claude -p "
You are AutoLab, an autonomous experiment runner for EOF3R Phase B.

Read these files first:
- outputs/autolab/state.json (current pipeline state)
- outputs/autolab/experiment_queue.yaml (pending experiments)
- outputs/autolab/ALERT.md (if exists, read it)

Then execute ONE cycle:
1. OBSERVE: Check GPU, disk, read existing results
2. ORIENT: Analyze what to do next based on state and queue
3. DECIDE: Pick the next pending experiment
4. ACT: Implement and run it (use conda activate eof3r + PYTHONPATH)
5. DOCUMENT: Update outputs/autolab/autolab_log.md with a ≤10 line summary

SAFETY RULES:
- Never exceed 90% GPU memory
- Never modify baselines/ source code
- Never pip install anything new
- All experiments go to outputs/autolab/results/<name>/
- If loss is NaN, kill the process immediately

After completing, update outputs/autolab/state.json with the new cycle number and results.
" 2>&1 | tee -a "$LOG_FILE"

    log "━━━ Cycle $cycle_num END ━━━"
}

# ============================================================
# MAIN LOOP
# ============================================================

log "============================================"
log "AutoLab Orchestrator Started"
log "PID: $$"
log "Project: $PROJECT_ROOT"
log "============================================"

# Initialize state if not exists
if [ ! -f "$OUTPUT_DIR/state.json" ]; then
    cat > "$OUTPUT_DIR/state.json" << 'EOF'
{
  "cycle": 0,
  "start_time": "2026-06-19T17:00:00",
  "experiments_run": [],
  "last_experiment": null,
  "consecutive_failures": 0,
  "baseline_metrics": null
}
EOF
    log "Initialized state.json"
fi

# Create autolab log if not exists
if [ ! -f "$OUTPUT_DIR/autolab_log.md" ]; then
    echo "# AutoLab Experiment Log" > "$OUTPUT_DIR/autolab_log.md"
    echo "" >> "$OUTPUT_DIR/autolab_log.md"
fi

cycle=0
while true; do
    cycle=$((cycle + 1))

    # Run cycle
    if ! run_cycle "$cycle"; then
        log "Pipeline halted — check ALERT.md"
        break
    fi

    # Wait for next cycle (1 hour)
    log "Sleeping until next cycle..."
    sleep 3600
done

log "AutoLab Orchestrator stopped"
rm -f "$PID_FILE"
