#!/bin/bash
folder="/scratch/ptah/anwe5599/CRIRES"
# day setting exptime
settings=(  "2022-11-29" "L3262" 30 \
            "2022-12-23" "M4318" 10 \
            "2022-12-23" "L3262" 30 \
            "2022-12-25" "L3340" 30 \
            "2022-12-31" "L3426" 60 \
            "2023-01-22" "M4318" 10 \
            "2023-02-15" "L3340" 60 \
            "2023-02-25" "M4318" 10 \
            "2023-02-26" "L3262" 60 \
        )
length=${#settings[@]}
length=$((length / 3))

for ((i=0;i<$length;i++)) do
    day=${settings[$i * 3]}
    wl=${settings[$i * 3 + 1]}
    exp=${settings[$i * 3 + 2]}
    echo $day\_$wl
    python make_sof.py $wl -r=$folder/$day\_$wl/raw -o=$folder/$day\_$wl/extr -e=$exp
done
