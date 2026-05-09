param([int]$MonitorPid, [int]$Crit = 85)

$log = "$PSScriptRoot\gpu_watchdog.log"
$marker = "$PSScriptRoot\watchdog_kill_marker.txt"

Add-Content $log "[$(Get-Date -Format HH:mm:ss)] Watchdog start PID=$MonitorPid crit=$Crit"

while (Get-Process -Id $MonitorPid -ErrorAction SilentlyContinue) {
    $temp = [int](nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | Select-Object -First 1)
    if ($temp -ge $Crit) {
        Add-Content $log "[$(Get-Date -Format HH:mm:ss)] CRITICAL $temp C -- killing PID=$MonitorPid"
        Stop-Process -Id $MonitorPid -Force -ErrorAction SilentlyContinue
        Set-Content $marker "KILLED_AT_${temp}C_$(Get-Date -Format HH:mm:ss)"
        exit 2
    }
    Start-Sleep -Seconds 30
}

Add-Content $log "[$(Get-Date -Format HH:mm:ss)] Watchdog done (training finished normally)"
exit 0
