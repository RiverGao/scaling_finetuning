for L in 1 2
do
  for region in angular middle superior inferior anterior
  do
    echo L$L $region
    python get_data_residues_fmri.py $L $region
  done
done