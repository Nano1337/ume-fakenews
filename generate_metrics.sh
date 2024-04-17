DATASET="weibo" # pick either "fakeddit" or "weibo"
MODEL_TYPE="qmf" # pick either "ensemble", "jlogits", "ogm_ge", "qmf"
CKPT="/home/haoli/Documents/ume-fakenews/data/weibo/_ckpts/weibo_cls2_qmf_chineseclip_lr=0.01_scheduler/scarlet-terrain-580_best.ckpt" # path to the checkpoint
python generate_metrics.py --dir $DATASET --model_type $MODEL_TYPE --ckpt $CKPT
