for model_residue in False
do
  for data_residue in False
  do
    for headpool in mean
    do
      for size in 30B
      do
        for lp in 1 2
        do
          for name in llama
          do
            echo $name $size $lp $model_residue $data_residue $headpool
            python llama_correlation_saccade.py $name $size $lp $model_residue $data_residue $headpool
          done
        done
      done
#      for view in number duration
#      do
#        echo $view $model_residue $data_residue $headpool
#        python show_saccade_correlation.py $view $model_residue $data_residue $headpool
#      done
    done
  done
done