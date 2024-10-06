for residue in True False
do
  echo residue: $residue
  echo size div
  python size_attn_div.py $residue

  for size in 7B 13B
  do
    echo size: $size, name div
    python name_attn_div.py $residue $size
  done

  for mission in name size
  do
    echo showing $mission div
    python show_name_part_divergence.py $residue $mission
  done
done