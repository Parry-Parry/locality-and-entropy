wget -nv --no-check-certificate https://rocketqa.bj.bcebos.com/V1/data_train.tar.gz
tar -zxf data_train.tar.gz --no-same-owner
rm -rf data_train.tar.gz

fold=marco
cat data_train/${fold}_de1_denoise.tsv data_train/${fold}_unlabel_de2_denoise.tsv > data_train/${fold}_merge_de2_denoise.tsv 
