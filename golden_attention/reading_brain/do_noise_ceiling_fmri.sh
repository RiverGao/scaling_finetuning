for L in 1 2
do
  for region in angular middle superior inferior anterior
  do
    for data_residue in False True
    do
      python compute_noise_ceiling_fmri.py $L $region $data_residue
    done
  done
done