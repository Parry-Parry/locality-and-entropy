wget -nv --no-check-certificate https://rocketqa.bj.bcebos.com/V1/data_train.tar.gz
tar -zxf data_train.tar.gz
rm -rf data_train.tar.gz

for fold in marco nq;do
    cat data_train/${fold}_de1_denoise.tsv data_train/${fold}_unlabel_de2_denoise.tsv > data_train/${fold}_merge_de2_denoise.tsv 
done