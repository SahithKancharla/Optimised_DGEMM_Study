# reproduce.ps1 -- clone-and-run harness for native Windows.
# Narrower coverage than reproduce.sh: Strassen/Dumas need OpenMP 3.0
# tasks, which MSVC's /openmp (2.0) can't build. So: prefer WSL if
# present (delegates to reproduce.sh, full coverage); else fall back to
# MSVC for classical DGEMM + the native P/E-core tools; else print
# install instructions.
#
# Usage: powershell -ExecutionPolicy Bypass -File .\reproduce.ps1 [-Full]

param([switch]$Full, [switch]$Avx512, [int]$Runs = 30, [switch]$Smoke)
if ($Smoke) { $Runs = 3 }   # wiring check only, NOT publication-grade

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

function Write-Section($s) {
    Write-Host ""
    Write-Host "== $s ==" -ForegroundColor Cyan
}

Write-Host "================================================================"
Write-Host " Optimised_DGEMM_Study reproducibility harness (native Windows)"
Write-Host "================================================================"

# -----------------------------------------------------------------
# Path 1: WSL, if present -- gives full coverage, so prefer it.
# -----------------------------------------------------------------
$wslAvailable = $false
try {
    $distros = & wsl.exe -l -q 2>$null
    if ($LASTEXITCODE -eq 0 -and $distros) { $wslAvailable = $true }
} catch { $wslAvailable = $false }

if ($wslAvailable) {
    Write-Section "WSL detected -- delegating to reproduce.sh for full coverage"
    # Not using 'wsl.exe wslpath -a' here: called from PowerShell (not
    # cmd/bash) it mangles backslashes in the argument. Default WSL2
    # drive mounts are always /mnt/<lowercase-drive>/... , so just
    # build that directly (a custom wsl.conf mount root is the one
    # case this won't handle).
    $drive = $PSScriptRoot.Substring(0,1).ToLower()
    $rest = $PSScriptRoot.Substring(2) -replace '\\','/'
    $linuxPath = "/mnt/$drive$rest"
    $fullArg = ""
    if ($Full) { $fullArg = "--full" }
    $fullArg = "$fullArg --runs $Runs"
    if ($Avx512) {
        Write-Host "AVX-512 pipeline: bash ./avx512/run_avx512.sh $fullArg"
        & wsl.exe bash -lc "cd '$linuxPath' && bash ./avx512/run_avx512.sh $fullArg"
        exit $LASTEXITCODE
    }
    Write-Host "Running: wsl bash -lc `"cd '$linuxPath' && bash ./reproduce.sh $fullArg`""
    & wsl.exe bash -lc "cd '$linuxPath' && bash ./reproduce.sh $fullArg"
    exit $LASTEXITCODE
}

Write-Host "No WSL distribution found -- falling back to the native-Windows/MSVC subset."
Write-Host "(For full coverage -- including the Strassen/Dumas rebuilds, which need"
Write-Host " gcc's OpenMP 3.0 task support -- install WSL2: 'wsl --install', then"
Write-Host " re-run this script.)"

# -----------------------------------------------------------------
# Path 2: MSVC (cl.exe), directly or via vswhere + vcvars64.bat.
# -----------------------------------------------------------------
function Find-ClCompiler {
    $cmd = Get-Command cl.exe -ErrorAction SilentlyContinue
    if ($cmd) { return @{ Direct = $true; VcVars = $null } }

    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (Test-Path $vswhere) {
        $vsPath = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath 2>$null
        if ($vsPath) {
            $vcvars = Join-Path $vsPath "VC\Auxiliary\Build\vcvars64.bat"
            if (Test-Path $vcvars) { return @{ Direct = $false; VcVars = $vcvars } }
        }
    }
    return $null
}

$cl = Find-ClCompiler
if (-not $cl) {
    Write-Host ""
    Write-Host "ERROR: Neither WSL nor a Visual Studio C++ toolchain (cl.exe) was found." -ForegroundColor Red
    Write-Host "Install one of:"
    Write-Host "  - WSL2 + Ubuntu (recommended, full coverage): 'wsl --install', reboot, re-run this script."
    Write-Host "  - Visual Studio Build Tools, 'Desktop development with C++' workload"
    Write-Host "    (narrower coverage -- classical DGEMM + the two native topology tools only)."
    exit 1
}

function Invoke-Build([string]$SourceRel, [string]$ExeOut, [string]$ExtraFlags = "") {
    $src = Join-Path $PSScriptRoot $SourceRel
    if ($cl.Direct) {
        & cl.exe /O2 /arch:AVX2 $ExtraFlags.Split(' ') $src /Fe:$ExeOut /nologo | Out-Null
    } else {
        $cmdLine = "call `"$($cl.VcVars)`" >nul && cl.exe /O2 /arch:AVX2 $ExtraFlags $src /Fe:$ExeOut /nologo"
        & cmd.exe /c $cmdLine | Out-Null
    }
    if ($LASTEXITCODE -ne 0) { throw "Build failed: $SourceRel" }
}

$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$out = "results\reproduce_$ts"
New-Item -ItemType Directory -Force -Path "$out\bin" | Out-Null
Write-Host "Output: $out\"

# System/run metadata -- same schema as reproduce.sh's run_info.csv, so
# Linux and Windows runs concatenate into one cross-machine table.
$os = Get-CimInstance Win32_OperatingSystem
$cpu = Get-CimInstance Win32_Processor | Select-Object -First 1
$ramGb = [Math]::Round($os.TotalVisibleMemorySize / 1MB)
$osName = "$($os.Caption) $($os.Version)"
$cpuModel = $cpu.Name.Trim()
$physCores = $cpu.NumberOfCores
$gitCommit = "n/a"; $gitStatus = "n/a"
if (Get-Command git -ErrorAction SilentlyContinue) {
    $gitCommit = (git rev-parse --short HEAD 2>$null)
    if ($gitCommit) {
        $dirty = (git status --porcelain 2>$null)
        $gitStatus = if ($dirty) { "dirty (local edits present)" } else { "clean" }
    } else { $gitCommit = "n/a (not a git checkout)" }
}
$clVersionLine = (& cmd.exe /c "cl.exe 2>&1" ) | Select-Object -First 1
@"
run_timestamp: $ts
hostname: $env:COMPUTERNAME
os: $osName
cpu_model: $cpuModel
physical_cores: $physCores
logical_cores: $([Environment]::ProcessorCount)
ram_gb: $ramGb
compiler: cl.exe
compiler_version: $clVersionLine
git_commit: $gitCommit
git_status: $gitStatus
mode: $(if ($Full) { 'full' } else { 'quick' })
"@ | Tee-Object -FilePath "$out\system_info.txt"

Write-Section "Building (MSVC /O2 /arch:AVX2)"
Invoke-Build "src\avx2\ladder\matmul_ladder.c" "$out\bin\matmul_ladder.exe" "/openmp"
Invoke-Build "topology\classify_cores_native.c" "$out\bin\classify_cores_native.exe"
Invoke-Build "topology\peak_scaling_native.c" "$out\bin\peak_scaling_native.exe"
Write-Host "Build OK."

Write-Section "Hardware sanity check (smallest possible AVX2+FMA+OpenMP run)"
$probe = & "$out\bin\matmul_ladder.exe" full 128 1 static 0 1 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: smoke run failed. Output:" -ForegroundColor Red
    Write-Host $probe
    Write-Host "Most likely cause: this CPU does not support AVX2+FMA (required"
    Write-Host "everywhere in this project -- see FINDINGS.md Part A)."
    exit 1
}
Write-Host "OK -- AVX2+FMA+OpenMP all work on this machine."

$nproc = [Environment]::ProcessorCount
$T = [Math]::Min($nproc, 8)
$N = if ($Full) { 2048 } else { 1024 }
$reps = if ($Full) { 10 } else { 5 }

Write-Section "1/3 Classical DGEMM: correctness + performance (matmul_ladder.c)"
# Project standard (PROTOCOL.md): n=$Runs independent process launches.
$samples = @()
for ($r = 1; $r -le $Runs; $r++) {
    $line = & "$out\bin\matmul_ladder.exe" full $N $T guided 0 $reps
    $f = $line -split ","
    $samples += [pscustomobject]@{ run = $r; gflops = [double]$f[-2]; relerr = [double]$f[-1] }
}
$samples | Export-Csv -NoTypeInformation -Path "$out\matmul_ladder_samples.csv"
$vals = [double[]]$samples.gflops
$mean = ($vals | Measure-Object -Average).Average
$sd = if ($vals.Count -gt 1) { [Math]::Sqrt((($vals | ForEach-Object { ($_ - $mean) * ($_ - $mean) }) | Measure-Object -Sum).Sum / ($vals.Count - 1)) } else { 0 }
$rng = New-Object System.Random 12345
$boot = New-Object 'System.Collections.Generic.List[double]'
for ($b = 0; $b -lt 10000; $b++) {
    $s = 0.0
    for ($i = 0; $i -lt $vals.Count; $i++) { $s += $vals[$rng.Next($vals.Count)] }
    $boot.Add($s / $vals.Count)
}
$boot.Sort()
$ciLo = [Math]::Round($boot[249], 3); $ciHi = [Math]::Round($boot[9749], 3)
$gflops = [Math]::Round($mean, 3)
$relerr = ($samples.relerr | Measure-Object -Maximum).Maximum
$cv = [Math]::Round(100 * $sd / $mean, 2)
Write-Host "  N=$N T=$T  n=$Runs  GFLOP/s: $gflops [95% CI $ciLo, $ciHi] CV $cv%   worst relerr: $relerr"

Write-Section "2/3 P-core/E-core classification (classify_cores_native.c) -- the exact tool paper Section 3.1 quotes"
Write-Host "Per-logical-CPU sustained FMA throughput (heterogeneous CPUs show a"
Write-Host "clear split; homogeneous CPUs will correctly show a flat line --"
Write-Host "both are valid, informative outcomes, not a pass/fail test):"
$results = @()
for ($cpu = 0; $cpu -lt $nproc; $cpu++) {
    $g = & "$out\bin\classify_cores_native.exe" 500000000 $cpu 2>$null
    $results += [pscustomobject]@{ cpu = $cpu; gflops = [double]$g }
}
$results | Format-Table -AutoSize
$results | Export-Csv -NoTypeInformation -Path "$out\core_classification.csv"

Write-Section "3/3 Pinned aggregate peak GFLOP/s across all logical CPUs (peak_scaling_native.c)"
$allCpus = (0..($nproc - 1)) -join " "
$peakOut = & "$out\bin\peak_scaling_native.exe" $allCpus.Split(" ")
$peakOut | Tee-Object -FilePath "$out\peak_scaling.csv"

Write-Host ""
Write-Host "================================================================"
Write-Host " SUMMARY  (full detail + raw CSVs in $out\)"
Write-Host "================================================================"
Write-Host "Classical DGEMM: relerr=$relerr   GFLOP/s=$gflops"
Write-Host "Core classification and aggregate peak GFLOP/s: see tables above / CSVs."

$header = "timestamp,hostname,os,kernel,cpu_model,physical_cores,logical_cores,ram_gb,compiler,compiler_version,git_commit,git_status,mode,dgemm_N,dgemm_T,dgemm_gflops,dgemm_relerr,strassen_gflops,strassen_relerr,strassen_pass,dumas_gflops,dumas_relerr,dumas_pass,dumas_cost_pct"
$mode = "$(if ($Full) { 'full' } else { 'quick' })-n$Runs"
$row = "$ts,$env:COMPUTERNAME,`"$osName`",n/a,`"$cpuModel`",$physCores,$([Environment]::ProcessorCount),$ramGb,cl.exe,`"$clVersionLine`",$gitCommit,$gitStatus,$mode,$N,$T,$gflops,$relerr,n/a,n/a,n/a,n/a,n/a,n/a,n/a"
"$header`n$row" | Tee-Object -FilePath "$out\run_info.csv" | Out-Null
if (-not (Test-Path "results\run_log.csv")) { $header | Out-File -FilePath "results\run_log.csv" -Encoding utf8 }
$row | Out-File -FilePath "results\run_log.csv" -Append -Encoding utf8
Write-Host ""
Write-Host "Run info: $out\run_info.csv (also appended to results\run_log.csv)"
Write-Host ""
Write-Host "What this script does NOT cover on native Windows (by design --"
Write-Host "needs gcc's OpenMP 3.0 task support; use reproduce.sh under WSL2):"
Write-Host "  - the classic-Strassen and Dumas-2024 rebuilds (paper Sections 4.3, 6)"
Write-Host "  - the numerical-accuracy sweep (paper Section 6.4)"
Write-Host "  - production-BLAS baseline comparisons (needs OpenBLAS/BLIS/MKL)"
