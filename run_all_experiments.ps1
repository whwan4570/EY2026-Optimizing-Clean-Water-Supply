# A/B/C/D x seed(42,7,2024) = 12 runs, then --report
# Purpose: decompose whether the LB +0.07 gain comes from "more folds / DRP tuning / both"
# Output: saved under runs/, then experiment_report.csv with diagnostics to confirm the setting that is consistently good across seed averages

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

$combos = @("A", "B", "C", "D")
$seeds = @(42, 7, 2024)

$total = $combos.Count * $seeds.Count
$n = 0
foreach ($c in $combos) {
    foreach ($s in $seeds) {
        $n++
        Write-Host "`n========== [$n/$total] combo=$c seed=$s ==========" -ForegroundColor Cyan
        python scripts\models\benchmark_model.py --combo $c --seed $s
        if ($LASTEXITCODE -ne 0) {
            Write-Host "Failed: combo=$c seed=$s" -ForegroundColor Red
            exit $LASTEXITCODE
        }
    }
}

Write-Host "`n========== Generating report ==========" -ForegroundColor Green
python scripts\models\benchmark_model.py --report

Write-Host "`nDone. Check runs\ and experiment_report.csv." -ForegroundColor Green
