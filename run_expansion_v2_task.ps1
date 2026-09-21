$ErrorActionPreference = "Stop"

$root = $PSScriptRoot
$venvPython = Join-Path $root ".venv-cuda\Scripts\python.exe"
$python = if (Test-Path -LiteralPath $venvPython) { $venvPython } else { "py" }
$manifest = Join-Path $root "configs\expansion_campaign_v2.json"
$statusPath = Join-Path $root "run_state\expansion_formal_v2_status.json"
$launcherPath = Join-Path $root "run_state\expansion_formal_v2_launcher.json"
$admissionScript = Join-Path $root "formal_campaign_admission.py"
$admissionPath = Join-Path $root "run_state\formal_v2_admission.json"
$logDir = Join-Path $root "logs\expansion\formal_v2"
$stdout = Join-Path $logDir "supervisor.stdout.log"
$stderr = Join-Path $logDir "supervisor.stderr.log"
$arguments = @(
    (Join-Path $root "run_expansion_supervisor.py"),
    "--manifest", $manifest,
    "--status", $statusPath,
    "--workers", "4",
    "--device", "cuda",
    "--resume"
)

New-Item -ItemType Directory -Force -Path $logDir | Out-Null
if (Test-Path -LiteralPath $statusPath) {
    $prior = Get-Content -LiteralPath $statusPath -Raw | ConvertFrom-Json
    if ($prior.status -eq "COMPLETED") {
        exit 0
    }
    if ($prior.status -eq "RUNNING" -and $prior.pid) {
        $existing = Get-CimInstance Win32_Process -Filter "ProcessId = $($prior.pid)" -ErrorAction SilentlyContinue
        if ($existing -and $existing.CommandLine -like "*run_expansion_supervisor.py*") {
            exit 0
        }
    }
}

# A resume is allowed only after the persisted semantic/provenance admission record passes.
& $python $admissionScript check-admission --admission $admissionPath
if ($LASTEXITCODE -ne 0) {
    throw "FORMAL_CAMPAIGN_ADMISSION_GATE failed; refusing to resume formal_v2"
}

$launcher = [ordered]@{
    schema_version = "1.0.0"
    status = "RUNNING"
    launch_mode = "WINDOWS_TASK_SCHEDULER"
    task_name = "CCTA-TTA-Expansion-Formal-V2"
    launcher_pid = $PID
    started_at = (Get-Date).ToUniversalTime().ToString("o")
    command = "$python $($arguments -join ' ')"
    stdout = $stdout
    stderr = $stderr
    status_path = $statusPath
}
$launcher | ConvertTo-Json -Depth 5 | Set-Content -Encoding utf8 $launcherPath

try {
    & $python @arguments 1>> $stdout 2>> $stderr
    $launcher.exit_code = $LASTEXITCODE
    $launcher.status = if ($LASTEXITCODE -eq 0) { "COMPLETED" } else { "FAILED" }
}
catch {
    $launcher.exit_code = 1
    $launcher.status = "LAUNCH_FAILED"
    $launcher.error = $_.Exception.Message
}
finally {
    $launcher.finished_at = (Get-Date).ToUniversalTime().ToString("o")
    $launcher | ConvertTo-Json -Depth 5 | Set-Content -Encoding utf8 $launcherPath
}

exit $launcher.exit_code
