$out = "cond2_unpinned_dynamic.csv"
"run,gflops" | Out-File -FilePath $out -Encoding utf8
for ($i = 1; $i -le 30; $i++) {
    $line = & .\matmul_ladder.exe full 1024 8 dynamic 4 5
    $fields = $line -split ","
    "$i,$($fields[11])" | Out-File -FilePath $out -Encoding utf8 -Append
    Write-Output "run $i : $($fields[11])"
}
