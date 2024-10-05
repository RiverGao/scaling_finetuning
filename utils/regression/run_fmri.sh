for region in angular anterior inferior middle superior
do
  for name in gpt2
  do
    for size in large
    do
      python heads_vs_fmri.py $name $size $region
    done
  done
done