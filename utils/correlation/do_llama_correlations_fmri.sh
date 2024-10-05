for model_residue in True
do
  for data_residue in True
  do
    for headpool in mean
    do
#      for size in 7B 13B
#      do
#        for lp in 1 2
#        do
#          for name in llama alpaca-lora vicuna
#          do
#            for region in angular middle superior inferior anterior
#            do
#              echo $name $size $lp $model_residue $data_residue $headpool $region
#              python llama_correlation_fmri.py $name $size $lp $model_residue $data_residue $headpool $region
#            done
#          done
#        done
#      done
      for region in angular middle superior inferior anterior
      do
        echo $model_residue $data_residue $headpool $region
        python show_fmri_correlation.py $model_residue $data_residue $headpool $region
      done
    done
  done
done