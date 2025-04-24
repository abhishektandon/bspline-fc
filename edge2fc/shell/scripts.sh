#train
python3 main.py --config configs/Template-BBDM.yaml --train --sample_at_start --save_top --gpu_ids 0 \
--resume_model path/to/model_ckpt --resume_optim path/to/optim_ckpt

#test
# python3 main.py --config configs/Template-BBDM.yaml --sample_to_eval --gpu_ids 0 \
# --resume_model path/to/model_ckpt --resume_optim path/to/optim_ckpt

## LPIPS
#python3 preprocess_and_evaluation.py -f LPIPS -s source/dir -t target/dir -n 1

## diversity
#python3 preprocess_and_evaluation.py -f diversity -s source/dir -n 1

## fidelity
#fidelity --gpu 0 --fid --input1 path1 --input2 path2