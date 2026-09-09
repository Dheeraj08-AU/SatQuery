# Moves the ad-hoc debug scripts and the old test suite into
# _deprecated_scripts/. They call the previous inference API and will now fail.
# Run once, from the repo root, AFTER extracting the update pack.
#
# Explicit file lists, not globs: tests/test_core.py is the NEW suite and must
# not be swept up with the old one.

$ErrorActionPreference = 'Stop'

if (-not (Test-Path '.\modules\model_registry.py')) {
    Write-Host 'ERROR: run this from the SatQuery-main repo root.' -ForegroundColor Red
    Write-Host "Current directory: $((Get-Location).Path)"
    exit 1
}

$dest = '.\_deprecated_scripts'
$oldTests = "$dest\old_tests"
foreach ($d in @($dest, $oldTests)) {
    if (-not (Test-Path $d)) { New-Item -ItemType Directory -Path $d | Out-Null }
}

$rootScripts = @(
    'test_allowlist_filter.py', 'test_area_filter.py', 'test_cache_hit_time.py',
    'test_change_analysis.py', 'test_change_prompts.py', 'test_e2e_pipeline.py',
    'test_false_positives.py', 'test_geojson_tif.py', 'test_geojson_verify.py',
    'test_memory_sequential.py', 'test_multiprobe.py', 'test_multiprobe_realistic.py',
    'test_multi_noun_evaluation.py', 'test_new_images_grounding.py', 'test_no_change.py',
    'test_optical_sar.py', 'test_partial_hallucinations.py', 'test_solo_nouns.py',
    'test_solo_urban.py', 'test_two_step_change.py', 'test_vqa_cpu_speed.py',
    'test_vqa_real.py', 'test_vqa_repetition.py',
    'run_real_sanity.py', 'run_real_sanity2.py', 'run_real_sanity_nouns.py',
    'run_road_tests.py', 'sanity_check_real_images.py', 'debug_grounding.py',
    'prewarm_cache.py', 'schema_check.py'
)

$testFiles = @(
    'conftest.py', 'test_agent.py', 'test_agent_controller_e2e.py',
    'test_agent_controller_real_llm.py', 'test_grounding.py'
)

$moved = 0
$skipped = 0

foreach ($name in $rootScripts) {
    $from = ".\$name"
    $to = "$dest\$name"
    if (-not (Test-Path $from)) { $skipped++; continue }
    if (Test-Path $to) {
        Write-Host "  skip (already there): $name" -ForegroundColor Yellow
        $skipped++
        continue
    }
    Move-Item -Path $from -Destination $to
    Write-Host "  moved: $name" -ForegroundColor Green
    $moved++
}

foreach ($name in $testFiles) {
    $from = ".\tests\$name"
    $to = "$oldTests\$name"
    if (-not (Test-Path $from)) { $skipped++; continue }
    if (Test-Path $to) {
        Write-Host "  skip (already there): tests\$name" -ForegroundColor Yellow
        $skipped++
        continue
    }
    Move-Item -Path $from -Destination $to
    Write-Host "  moved: tests\$name" -ForegroundColor Green
    $moved++
}

Write-Host ''
Write-Host "Done. $moved moved, $skipped skipped (already moved or absent)."

if (Test-Path '.\tests\test_core.py') {
    Write-Host ''
    Write-Host 'New unit suite is in place. Verify with:' -ForegroundColor Cyan
    Write-Host '    pytest tests -q' -ForegroundColor Cyan
    Write-Host 'It needs no model weights, no GPU and no network.' -ForegroundColor Cyan
} else {
    Write-Host ''
    Write-Host 'WARNING: tests\test_core.py is missing - extract the update pack first.' -ForegroundColor Red
}
