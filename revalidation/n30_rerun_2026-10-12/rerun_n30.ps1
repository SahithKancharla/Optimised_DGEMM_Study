# rerun_n30.ps1 -- n=30-repetition re-run of the Section 5 (paper) / Part G.4
# (FINDINGS.md) controlled core-placement experiment, native Windows/MSVC,
# matmul_ladder.exe, N=1024 T=8 chunk=4 reps=5(internal best-of) per
# invocation -- exactly matching pinning_experiment.csv's original
# configuration, just with 30 independent process invocations per
# condition instead of 4-5, so a real bootstrap CI / significance test
# can be computed instead of eyeballing a reported range.
#
# Outlier-retry note (found while building this script, 2026-10-12):
# intermittent, unexplained catastrophic slowdowns (observed down to
# ~0.2-0.7 GFLOP/s, vs. a healthy ~190-300 GFLOP/s band, at the SAME
# N/T/schedule) occur occasionally and non-deterministically on this
# machine -- reproduced with byte-identical invocations run moments
# apart, one healthy, the next degraded ~400x. Not attributable to any
# bug found in this script, to argument-passing, to the exe itself (the
# correctness check (relerr) still reports 0.000e+00 even on a degraded
# run -- the result is correct, just slow), to WSL2 (reproduced with
# WSL2 fully shut down), or to thermal/sustained-load buildup (a clean
# 120-iteration single-condition run showed zero degradation; degraded
# runs show it from iteration 1, not building up). Root cause not
# identified. Below: a sanity floor (50 GFLOP/s, well under every
# legitimate measurement anywhere in this paper for this N/T) triggers
# an automatic retry (up to 5x) rather than silently accepting or
# silently discarding an anomalous reading.
$exe = ".\matmul_ladder.exe"
$N = 1024
$T = 8
$reps = 5
$n = 30
$floor = 50.0
$maxRetries = 5
$out = "rerun_n30_results.csv"
$anomalyLog = "rerun_n30_anomalies.csv"

$conditions = @(
    @{label="unpinned_static";  sched="static";  chunk=0; cores=""},
    @{label="unpinned_dynamic"; sched="dynamic"; chunk=4; cores=""},
    @{label="mixed_PE_static";  sched="static";  chunk=0; cores="0,2,4,6,16,17,18,19"},
    @{label="mixed_PE_dynamic"; sched="dynamic"; chunk=4; cores="0,2,4,6,16,17,18,19"}
)

"label,run,gflops" | Out-File -FilePath $out -Encoding utf8
"label,run,attempt,gflops_rejected" | Out-File -FilePath $anomalyLog -Encoding utf8

foreach ($c in $conditions) {
    $sched = $c.sched
    $chunk = $c.chunk
    $cores = $c.cores
    for ($i = 1; $i -le $n; $i++) {
        $gflops = 0.0
        for ($attempt = 1; $attempt -le $maxRetries; $attempt++) {
            if ($cores -ne "") {
                $line = & $exe full $N $T $sched $chunk $reps $cores
            } else {
                $line = & $exe full $N $T $sched $chunk $reps
            }
            $fields = $line -split ","
            $gflops = [double]$fields[11]
            if ($gflops -ge $floor) { break }
            "$($c.label),$i,$attempt,$gflops" | Out-File -FilePath $anomalyLog -Encoding utf8 -Append
            Write-Host "  [anomaly] $($c.label) run $i attempt $attempt : $gflops GFLOP/s (below floor, retrying)"
        }
        "$($c.label),$i,$gflops" | Out-File -FilePath $out -Encoding utf8 -Append
        Write-Host "$($c.label) run $i : $gflops GFLOP/s"
    }
}
Write-Host "Done -- wrote $out and $anomalyLog"
