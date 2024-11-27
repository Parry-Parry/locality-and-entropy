mkdir data_train
cd data_train
for filename in marco_joint.rand8.tar.gz marco_joint.rand128-part0.tar.gz marco_joint.rand128-part1.tar.gz marco_joint.aug128-part0.tar.gz marco_joint.aug128-part1.tar.gz nq_joint.rand32+aug32.tar.gz nq_joint.rand8.tar.gz;do
    wget -nv --no-check-certificate https://rocketqa.bj.bcebos.com/V2/data_train/$filename
    tar -xzf $filename
    rm -rf $filename
done
cat marco_joint.rand128-part0 marco_joint.rand128-part1 marco_joint.aug128-part0 marco_joint.aug128-part1 > marco_joint.rand128+aug128
rm -rf marco_joint.rand128-part0 marco_joint.rand128-part1 marco_joint.aug128-part0 marco_joint.aug128-part1
