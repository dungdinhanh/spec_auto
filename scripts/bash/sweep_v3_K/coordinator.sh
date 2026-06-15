#!/bin/bash
# K-sweep coordinator. Runs after Option B finishes.
# Phase 1 (parallel on 8 GPUs): K=10 on 0-3, K=15 on 4-7
# Phase 2 (4 GPUs): K=20 on 0-3
# Each training is followed by its own eval.
set -u
LOG=/tmp/ksweep_coord.log
echo "coordinator started $(date)" > $LOG

# --- Wait for any prior trainings/evals (filter out self via $$) ---
# pgrep matches command lines, so we exclude our own PID and filter to python processes only.
echo "[1] waiting for prior trainings/evals..." | tee -a $LOG
self_pid=$$
count_train() {
  pgrep -f train_dflash_rl_action_v3.py 2>/dev/null \
    | grep -v "^${self_pid}$" \
    | xargs -I{} ps -p {} -o cmd= 2>/dev/null \
    | grep -E '^[^ ]*python.*train_dflash_rl_action_v3.py' | wc -l
}
count_eval() {
  pgrep -f eval_ckpt_sweep_vt 2>/dev/null \
    | grep -v "^${self_pid}$" \
    | xargs -I{} ps -p {} -o cmd= 2>/dev/null \
    | grep -E '^[^ ]*python.*eval_ckpt_sweep_vt' | wc -l
}
while [ "$(count_train)" -gt 0 ]; do
  echo "$(date) prior train alive ($(count_train) python procs)" >> $LOG
  sleep 120
done
echo "[1] prior trainings ended $(date)" | tee -a $LOG
sleep 30
while [ "$(count_eval)" -gt 0 ]; do
  echo "$(date) prior eval alive ($(count_eval) python procs)" >> $LOG
  sleep 60
done
echo "[1] prior evals ended $(date)" | tee -a $LOG

# --- Phase 1: K=10 + K=15 in parallel ---
echo "[2] launching K=10 (GPUs 0-3) and K=15 (GPUs 4-7)" | tee -a $LOG
nohup bash /home/ubuntu/sweep_v3_K/launch_K10.sh > /tmp/ksweep_K10.log 2>&1 &
PID_K10=$!
sleep 5
nohup bash /home/ubuntu/sweep_v3_K/launch_K15.sh > /tmp/ksweep_K15.log 2>&1 &
PID_K15=$!
echo "K10_pid=$PID_K10 K15_pid=$PID_K15 at $(date)" | tee -a $LOG

# Wait for K=10 to finish, then start its eval
wait $PID_K10
echo "[2] K=10 finished $(date)" | tee -a $LOG
bash /home/ubuntu/sweep_v3_K/eval.sh K10 0 1 2 3 >> $LOG 2>&1 &
PID_E10=$!

# Wait for K=15 to finish, then start its eval
wait $PID_K15
echo "[2] K=15 finished $(date)" | tee -a $LOG
# Wait for K=10 eval to finish before grabbing its GPUs (we'll need them for K=20)
wait $PID_E10
echo "[2] K=10 eval finished $(date)" | tee -a $LOG
bash /home/ubuntu/sweep_v3_K/eval.sh K15 4 5 6 7 >> $LOG 2>&1 &
PID_E15=$!

# --- Phase 2: K=20 on GPUs 0-3 (in parallel with K=15 eval on 4-7) ---
echo "[3] launching K=20 (GPUs 0-3)" | tee -a $LOG
nohup bash /home/ubuntu/sweep_v3_K/launch_K20.sh > /tmp/ksweep_K20.log 2>&1 &
PID_K20=$!
echo "K20_pid=$PID_K20 at $(date)" | tee -a $LOG

wait $PID_E15
echo "[3] K=15 eval finished $(date)" | tee -a $LOG
wait $PID_K20
echo "[3] K=20 finished $(date)" | tee -a $LOG
bash /home/ubuntu/sweep_v3_K/eval.sh K20 0 1 2 3 >> $LOG 2>&1
echo "[3] K=20 eval finished $(date)" | tee -a $LOG

echo "ALL K-SWEEP DONE $(date)" | tee -a $LOG
