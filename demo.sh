DATASET="fakeddit" 
MODEL_TYPE="ensemble"
CKPT="/home/haoli/Documents/ume-fakenews/data/fakeddit/_ckpts/fakeddit_cls2_ensemble_sgd_scheduler/smart-morning-551_best.ckpt" 
python demo.py --dir $DATASET --model_type $MODEL_TYPE --ckpt $CKPT
