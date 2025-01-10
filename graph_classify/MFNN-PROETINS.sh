#!/bin/bash

python -m run_tu_exp \
  --device 0 \
  --model 'MFNN' \
  --exp_name MFNN \
  --dataset PROTEINS \
  --lr 0.005 \
  --batch_size 32 \
  --drop_rate 0.0 \
  --emb_dim 32 \
  --readout sum \
  --lr_scheduler 'StepLR' \
  --lr_scheduler_decay_steps 50 \
  --lr_scheduler_decay_rate 0.5 \
  --final_readout sum \
  --init_method mean \
  --jump_mode 'cat' \
  --graph_norm bn \
  --use_coboundaries False \
  --dump_curves \
  --preproc_jobs 4 \
  --petalType 'simplex' \
  --max_petal_dim 2 \
  --train_eval_period 50 \
  --epochs 200 \
  --num_layers 1 \
  --drop_position 'final_readout' \
  --task_type classification \
  --eval_metric accuracy \
  --max_dim 2 \
  --nonlinearity relu \

python -m run_tu_exp \
  --device 0 \
  --model 'MFNN' \
  --exp_name MFNN \
  --dataset PTC \
  --lr 0.005 \
  --batch_size 32 \
  --drop_rate 0.0 \
  --emb_dim 32 \
  --readout sum \
  --lr_scheduler 'StepLR' \
  --lr_scheduler_decay_steps 50 \
  --lr_scheduler_decay_rate 0.5 \
  --final_readout sum \
  --init_method mean \
  --jump_mode 'cat' \
  --graph_norm bn \
  --use_coboundaries False \
  --dump_curves \
  --preproc_jobs 4 \
  --petalType 'simplex' \
  --max_petal_dim 2 \
  --train_eval_period 50 \
  --epochs 200 \
  --num_layers 1 \
  --drop_position 'final_readout' \
  --task_type classification \
  --eval_metric accuracy \
  --max_dim 2 \
  --nonlinearity relu \

python -m run_tu_exp \
  --device 0 \
  --model 'MFNN' \
  --exp_name MFNN \
  --dataset MUTAG \
  --lr 0.005 \
  --batch_size 32 \
  --drop_rate 0.0 \
  --emb_dim 32 \
  --readout sum \
  --lr_scheduler 'StepLR' \
  --lr_scheduler_decay_steps 50 \
  --lr_scheduler_decay_rate 0.5 \
  --final_readout sum \
  --init_method mean \
  --jump_mode 'cat' \
  --graph_norm bn \
  --use_coboundaries False \
  --dump_curves \
  --preproc_jobs 4 \
  --petalType 'simplex' \
  --max_petal_dim 2 \
  --train_eval_period 50 \
  --epochs 200 \
  --num_layers 1 \
  --drop_position 'final_readout' \
  --task_type classification \
  --eval_metric accuracy \
  --max_dim 2 \
  --nonlinearity relu \



#python -m run_tu_exp \
#  --device 0 \
#  --model 'MFNN' \
#  --exp_name MFNN \
#  --dataset IMDBBINARY \
#  --lr 0.005 \
#  --batch_size 32 \
#  --drop_rate 0.0 \
#  --emb_dim 32 \
#  --readout sum \
#  --lr_scheduler 'StepLR' \
#  --lr_scheduler_decay_steps 50 \
#  --lr_scheduler_decay_rate 0.5 \
#  --final_readout sum \
#  --init_method mean \
#  --jump_mode 'cat' \
#  --graph_norm bn \
#  --use_coboundaries False \
#  --dump_curves \
#  --preproc_jobs 4 \
#  --petalType 'simplex' \
#  --max_petal_dim 2 \
#  --train_eval_period 50 \
#  --epochs 200 \
#  --num_layers 1 \
#  --drop_position 'final_readout' \
#  --task_type classification \
#  --eval_metric accuracy \
#  --max_dim 2 \
#  --nonlinearity relu \
#
#
#python -m run_tu_exp \
#  --device 0 \
#  --model 'MFNN' \
#  --exp_name MFNN \
#  --dataset IMDBMULTI \
#  --lr 0.005 \
#  --batch_size 32 \
#  --drop_rate 0.0 \
#  --emb_dim 32 \
#  --readout sum \
#  --lr_scheduler 'StepLR' \
#  --lr_scheduler_decay_steps 50 \
#  --lr_scheduler_decay_rate 0.5 \
#  --final_readout sum \
#  --init_method mean \
#  --jump_mode 'cat' \
#  --graph_norm bn \
#  --use_coboundaries False \
#  --dump_curves \
#  --preproc_jobs 4 \
#  --petalType 'simplex' \
#  --max_petal_dim 2 \
#  --train_eval_period 50 \
#  --epochs 200 \
#  --num_layers 1 \
#  --drop_position 'final_readout' \
#  --task_type classification \
#  --eval_metric accuracy \
#  --max_dim 2 \
#  --nonlinearity relu \
#

