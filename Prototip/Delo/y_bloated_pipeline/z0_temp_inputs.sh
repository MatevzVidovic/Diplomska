#!/bin/bash

# Creates temp files save_and_stop, results_and_stop, and graph_and_stop

save_and_stop=$(mktemp)
printf "s\nstop\n" > "$save_and_stop"
# python3 ${main_file} --ips 0 --sd ${folder_name}  -t < "$save_and_stop"           2>&1 | tee "${rfn}/curr/${obn}_${cbi}_${cn}.txt"; cn=$((cn + 1))


results_and_stop=$(mktemp)
printf "r\nstop\n" > "$results_and_stop"
# python3 ${main_file} --ips 0 --sd ${folder_name}  -t < "$results_and_stop"           2>&1 | tee "${rfn}/curr/${obn}_${cbi}_${cn}.txt"; cn=$((cn + 1))

graph_and_stop=$(mktemp)
printf "g\nstop\n" > "$graph_and_stop"
# python3 ${main_file} --ips 0 --sd ${folder_name}  -t < "$graph_and_stop"           2>&1 | tee "${rfn}/curr/${obn}_${cbi}_${cn}.txt"; cn=$((cn + 1))

resource_graph_and_stop=$(mktemp)
printf "resource_graph\nstop\n" > "$resource_graph_and_stop"
