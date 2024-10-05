for name in vicuna-old
do
  for size in 7B 13B
  do
    echo $name $size
    python get_model_residues_rb.py $name $size
    python get_model_residues_amr.py $name $size
  done
done