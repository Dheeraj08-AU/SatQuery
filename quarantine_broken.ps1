# Moves the broken training scripts and degenerate datasets into
# training/_deprecated/. Run once, from the repo root.
#
# Nothing is deleted. Anything already moved is skipped. Existing
# destinations are never overwritten.

$ErrorActionPreference = 'Stop'

if (-not (Test-Path '.\modules\model_registry.py')) {
    Write-Host 'ERROR: run this from the SatQuery-main repo root.' -ForegroundColor Red
    Write-Host "Current directory: $((Get-Location).Path)"
    exit 1
}

$dest = '.\training\_deprecated'
if (-not (Test-Path $dest)) {
    New-Item -ItemType Directory -Path $dest | Out-Null
    Write-Host "created $dest"
}

$moves = @(
    @{ From = '.\training\finetune_vlm.py';        To = "$dest\finetune_vlm.py" },
    @{ From = '.\training\finetune_change_vlm.py'; To = "$dest\finetune_change_vlm.py" },
    @{ From = '.\training\cdvqa_formatted';        To = "$dest\cdvqa_formatted_BROKEN" },
    @{ From = '.\data\vrsbench_train.json';        To = "$dest\vrsbench_train_BROKEN.json" }
)

$moved   = 0
$skipped = 0

foreach ($m in $moves) {
    $from = $m.From
    $to   = $m.To

    if (-not (Test-Path $from)) {
        Write-Host "  skip (not present): $from" -ForegroundColor DarkGray
        $skipped++
        continue
    }
    if (Test-Path $to) {
        Write-Host "  skip (destination exists): $to" -ForegroundColor Yellow
        $skipped++
        continue
    }

    Move-Item -Path $from -Destination $to
    Write-Host "  moved: $from  ->  $to" -ForegroundColor Green
    $moved++
}

Write-Host ''
Write-Host "Done. $moved moved, $skipped skipped."
Write-Host "See training\_deprecated\README.md for why each item was quarantined."

# Reminder about the one thing an archive cannot carry.
if (Test-Path '.\modules\satquery_vqa_adapter\adapter_model.safetensors') {
    Write-Host ''
    Write-Host 'Your VQA adapter weights are present on this machine.' -ForegroundColor Cyan
    Write-Host 'Push them to a private HF repo before retraining, so the current' -ForegroundColor Cyan
    Write-Host 'state is recoverable if the new run goes badly.' -ForegroundColor Cyan
}
