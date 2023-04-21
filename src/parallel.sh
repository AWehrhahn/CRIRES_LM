#!/bin/bash

parallel --bar --joblog parallel.log --results log/ "cd /scratch/ptah/anwe5599/CRIRES/{}/extr; ./reduce_cals.sh" ::: 2022-11-29_L3262 2022-12-23_M4318 2022-12-23_L3262 2022-12-25_L3340 2022-12-31_L3426 2023-01-22_M4318 2023-02-15_L3340 2023-02-25_M4318 2023-02-26_L3262