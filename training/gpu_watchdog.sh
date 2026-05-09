#!/bin/bash
# GPU watchdog — har 30s temperaturani kuzatadi
# Agar >= CRIT °C bo'lsa, berilgan PID ni o'ldiradi
# Usage: ./gpu_watchdog.sh <training_pid>

PID="$1"
CRIT=85
LOG=gpu_watchdog.log

if [ -z "$PID" ]; then
    echo "Usage: $0 <training_pid>"
    exit 1
fi

echo "[$(date +%H:%M:%S)] Watchdog start, monitoring PID=$PID, crit=${CRIT}°C" >> "$LOG"

while kill -0 "$PID" 2>/dev/null; do
    temp=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | tr -d ' \r')
    if [ "$temp" -ge "$CRIT" ]; then
        echo "[$(date +%H:%M:%S)] CRITICAL ${temp}°C — killing PID=$PID" >> "$LOG"
        # Windows process kill
        taskkill //F //PID "$PID" 2>&1 | head -3 >> "$LOG"
        echo "KILLED_AT_${temp}C" > /tmp/watchdog_kill_marker
        exit 2
    fi
    sleep 30
done

echo "[$(date +%H:%M:%S)] Watchdog done, training finished normally" >> "$LOG"
exit 0
