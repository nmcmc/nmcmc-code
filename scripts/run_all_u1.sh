#!/usr/bin/env sh
# Run all scripts in the scripts directory

if [ $# -eq 0 ]; then
    dir=$(pwd)/scripts
else
    dir=$1
fi


f="$dir/u1.py --equiv plaq --coupling ncp --loss rt --verbose 1 --n-eras  1 --n-samples  4096"
echo "Running $f"
$f
f="$dir/u1.py --equiv 2x1  --coupling ncp --loss rt --verbose 1 --n-eras  1 --n-samples  4096"
echo "Running $f"
$f
f="$dir/u1.py --equiv sch  --coupling ncp --loss rt --verbose 1 --n-eras  1 --n-samples  4096"
echo "Running $f"
$f

f="$dir/u1.py --equiv plaq --coupling cs --loss rt --verbose 1 --n-eras  1 --n-samples  4096"
echo "Running $f"
$f
f="$dir/u1.py --equiv 2x1  --coupling cs --loss rt --verbose 1 --n-eras  1 --n-samples  4096"
echo "Running $f"
$f
f="$dir/u1.py --equiv sch  --coupling cs --loss rt --verbose 1 --n-eras  1 --n-samples  4096"
echo "Running $f"
$f


f="$dir/u1.py --equiv plaq --coupling cs --loss REINFORCE --verbose 1 --n-eras  1 --n-samples  4096"
echo "Running $f"
$f
f="$dir/u1.py --equiv 2x1  --coupling cs --loss REINFORCE --verbose 1 --n-eras  1 --n-samples  4096"
echo "Running $f"
$f
f="$dir/u1.py --equiv sch  --coupling cs --loss REINFORCE --verbose 1 --n-eras  1 --n-samples  4096"
echo "Running $f"
$f




