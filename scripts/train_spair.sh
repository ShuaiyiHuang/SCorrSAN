prefix='scorrsan'
exp_id=0
epoch=50
benchmark='spair'
sce_ksize=7
sce_outdim=2048
remember_rate=0.2
num_grad=10
pseudo_ksize=7
lmd_loss_pseudo=10.0
pstart_ep=4

source activate scorrsan

python train.py --name_exp 2022${prefix}_exp${exp_id}_${benchmark}_k${sce_ksize}-outd${sce_outdim}_rr${remember_rate}_ng${num_grad}_ps${pseudo_ksize}_lp${lmd_loss_pseudo}_psep${pstart_ep} --benchmark ${benchmark}\
    --sce_ksize ${sce_ksize} --sce_outdim ${sce_outdim} --n_threads 4\
    --remember_rate ${remember_rate} --num_grad ${num_grad} \
    --pseudo_ksize ${pseudo_ksize} --lmd_loss_pseudo ${lmd_loss_pseudo} --pstart_ep ${pstart_ep} \

