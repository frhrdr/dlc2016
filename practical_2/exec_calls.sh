#!/bin/bash
#echo "ex-1"
#python train_mlp.py --learning_rate 1e-3 --weight_init normal --weight_init_scale 1e-5 --weight_reg none --log_dir ./logs/cifar10/inits/norm5
#python train_mlp.py --learning_rate 1e-3 --weight_init normal --weight_init_scale 1e-4 --weight_reg none --log_dir ./logs/cifar10/inits/norm4
#python train_mlp.py --learning_rate 1e-3 --weight_init normal --weight_init_scale 1e-3 --weight_reg none --log_dir ./logs/cifar10/inits/norm3
#python train_mlp.py --learning_rate 1e-3 --weight_init normal --weight_init_scale 1e-2 --weight_reg none --log_dir ./logs/cifar10/inits/norm2
#python train_mlp.py --learning_rate 1e-3 --weight_init uniform --weight_init_scale 2e-5 --weight_reg none --log_dir ./logs/cifar10/inits/unif5
#python train_mlp.py --learning_rate 1e-3 --weight_init uniform --weight_init_scale 2e-4 --weight_reg none --log_dir ./logs/cifar10/inits/unif4
#python train_mlp.py --learning_rate 1e-3 --weight_init uniform --weight_init_scale 2e-3 --weight_reg none --log_dir ./logs/cifar10/inits/unif3
#python train_mlp.py --learning_rate 1e-3 --weight_init uniform --weight_init_scale 2e-2 --weight_reg none --log_dir ./logs/cifar10/inits/unif2
#python train_mlp.py --learning_rate 1e-3 --weight_init xavier --weight_reg none --log_dir ./logs/cifar10/inits/xavier
#echo "ex-2"
#python train_mlp.py --activation tanh --learning_rate 1e-3 --weight_init normal --weight_init_scale 0.001 --weight_reg none --log_dir ./logs/cifar10/inacts/tanhnorm
#python train_mlp.py --activation tanh --learning_rate 1e-3 --weight_init xavier --weight_reg none --log_dir ./logs/cifar10/inacts/tanhxavier
#python train_mlp.py --activation relu --learning_rate 1e-3 --weight_init normal --weight_init_scale 0.001 --weight_reg none --log_dir ./logs/cifar10/inacts/relunorm
#python train_mlp.py --activation relu --learning_rate 1e-3 --weight_init xavier --weight_init_scale 0. --weight_reg none --log_dir ./logs/cifar10/inacts/reluxavier
#echo "ex-3"
#python train_mlp.py --learning_rate 1e-3 --dnn_hidden_units 300,300 --activation tanh --weight_init normal --weight_init_scale 0.001 --weight_reg none --log_dir ./logs/cifar10/arch/tanhnorm
#python train_mlp.py --learning_rate 1e-3 --dnn_hidden_units 300,300 --activation tanh --weight_init xavier --weight_init_scale 0.001 --weight_reg none --log_dir ./logs/cifar10/arch/tanhxavier
#python train_mlp.py --learning_rate 1e-3 --dnn_hidden_units 300,300 --activation relu --weight_init normal --weight_init_scale 0.001 --weight_reg none --log_dir ./logs/cifar10/arch/relunorm
#python train_mlp.py --learning_rate 1e-3 --dnn_hidden_units 300,300 --activation relu --weight_init xavier --weight_init_scale 0.001 --weight_reg none --log_dir ./logs/cifar10/arch/reluxavier
#echo "ex-4"
#python train_mlp.py --learning_rate 1e-3 --dnn_hidden_units 300,300 --activation relu --weight_init normal --weight_init_scale 1e-3 --optimizer adam --weight_reg none --log_dir ./logs/cifar10/opts/adam
#python train_mlp.py --learning_rate 1e-3 --dnn_hidden_units 300,300 --activation relu --weight_init normal --weight_init_scale 1e-3 --optimizer adagrad --weight_reg none --log_dir ./logs/cifar10/opts/adagrad
#python train_mlp.py --learning_rate 1e-3 --dnn_hidden_units 300,300 --activation relu --weight_init normal --weight_init_scale 1e-3 --optimizer adadelta --weight_reg none --log_dir ./logs/cifar10/opts/adadelta
#python train_mlp.py --learning_rate 1e-3 --dnn_hidden_units 300,300 --activation relu --weight_init normal --weight_init_scale 1e-3 --optimizer rmsprop --weight_reg none --log_dir ./logs/cifar10/opts/rmsprop
#python train_mlp.py --learning_rate 1e-3 --dnn_hidden_units 300,300 --activation relu --weight_init normal --weight_init_scale 1e-3 --optimizer sgd --weight_reg none --log_dir ./logs/cifar10/opts/sgd
#echo "ex-5"
python train_mlp.py --dnn_hidden_units 800 --learning_rate 1e-3 --weight_reg l2 --max_steps 3500 --batch_size 200 --dropout_rate 0. --activation elu --weight_init_scale 1e-3 --weight_init normal --optimizer adagrad --dropout_rate 0.5 --weight_reg_strength 0.2 --log_dir ./logs/cifar10/tiptop
#python train_mlp.py --learning_rate 1e-3 --dnn_hidden_units 600,400 --activation relu --weight_init xavier --weight_init_scale 1e-3 --optimizer adagrad --weight_reg l2 --weight_reg_scale 0.01 --dropout_rate 0.5 --log_dir ./logs/cifar10/tiptop
