DATATSET="fakeddit" # pick either "fakeddit" or "weibo"
MODEL_TYPE="qmf" # pick either "ensemble", "jlogits", "ogm_ge", "qmf"
CKPT="/home/haoli/Documents/ume-fakenews/data/fakeddit/_ckpts/fakeddit_cls2_qmf_sgd_scheduler/smooth-river-556_best.ckpt" # path to the checkpoint
python generate_metrics.py --dir $DATATSET --model_type $MODEL_TYPE --ckpt $CKPT

