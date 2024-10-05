for residue in True False
do
  for headpool in mean
  do
    for size in 30B
    do
      for name in llama
      do
#        echo amr correlation: $name $size $residue $headpool
#        python llama_correlation_amr.py $name $size $residue $headpool

        for task in predicate
        do
          echo $task correlation: $size $residue $headpool
          python llama_correlation_labels.py $name $size $residue $headpool $task
        done
      done
    done

    echo showing: $residue $headpool
#    python show_amr_correlation.py $residue $headpool
    python show_label_correlation.py $residue $headpool
  done
done