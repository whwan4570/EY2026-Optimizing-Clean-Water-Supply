# A/B/C/D x seed(42,7,2024) = 12 runs, then --report
# 목적: LB +0.07 개선이 "fold 증가 / DRP 튜닝 / 둘 다" 중 어디서 오는지 분해
# 결과: runs/ 에 저장 후 experiment_report.csv + 진단 문구로 "seed 평균에서 일관되게 좋은 설정" 확인

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
        python benchmark_model.py --combo $c --seed $s
        if ($LASTEXITCODE -ne 0) {
            Write-Host "실패: combo=$c seed=$s" -ForegroundColor Red
            exit $LASTEXITCODE
        }
    }
}

Write-Host "`n========== 리포트 생성 ==========" -ForegroundColor Green
python benchmark_model.py --report

Write-Host "`n완료. runs\ 및 experiment_report.csv 확인." -ForegroundColor Green
