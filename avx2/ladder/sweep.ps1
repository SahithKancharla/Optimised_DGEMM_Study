param([string]$Bin, [string]$EnvLabel)
"env,binary,stage,N,T,schedule,chunk,reps,pack_a_s,pack_b_s,kernel_s,total_s,gflops,relerr"
foreach ($N in 512,1024,2048) {
    foreach ($T in 1,4,8) {
        foreach ($SCHED in "static","dynamic","guided") {
            $CHUNK = 4
            if ($SCHED -eq "static" -or $SCHED -eq "guided") { $CHUNK = 0 }
            $line = & $Bin full $N $T $SCHED $CHUNK 5
            "$EnvLabel,$line"
        }
    }
}
