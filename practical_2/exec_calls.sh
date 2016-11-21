#!/bin/bash
python train_mlp.py --dnn_hidden_units 400 --learning_rate 1e-3 --weight_reg l2 \
--max_steps 3500 --batch_size 200 --dropout_rate 0. --activation elu --weight_init_scale 1e-3 \
--weight_init normal --optimizer adagrad --weight_reg_strength 0.2 --log_dir './logs/cifar10/topmodel'