$out = "cond4_mixed_PE_dynamic.csv"
"run,gflops" | Out-File -FilePath $out -Encoding utf8
for ($i = 1; $i -le 30; $i++) {
    $line = & .\matmul_ladder.exe full 1024 8 dynamic 4 5 0,2,4,6,16,17,18,19
    $fields = $line -split ","
    "$i,$($fields[11])" | Out-File -FilePath $out -Encoding utf8 -Append
    Write-Output "run $i : $($fields[11])"
}
