for name in llama alpaca vicuna-v0 vicuna-v1.1
do
  for size in 7B 13B
  do
    echo $name $size
    python get_model_residues_rb.py $name $size
  done
done