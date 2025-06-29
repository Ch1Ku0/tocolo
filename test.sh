lang=java
for project_name in termux; do
    mkdir -p ./saved_models/$lang
    python run_similarity_code.py \
        --output_dir=./saved_models/$project_name \
        --model_name_or_path=/home/chikuo/pretrain_model/model_repo/unixcoder/ \
        --do_test \
        --train_data_file=data/$project_name/${project_name}_train.json \
        --eval_data_file=data/$project_name/${project_name}_list_test.json \
        --codebase_file=data/$project_name/${project_name}_code.json \
        --num_train_epochs 30 \
        --code_length 256 \
        --nl_length 128 \
        --train_batch_size 16 \
        --eval_batch_size 32 \
        --learning_rate 2e-5 \
        --project_name $project_name \
        --result_file saved_models/${project_name}_result_unixcoder.json \
        --seed 1234567 2>&1| tee saved_models/test_${project_name}.log
done

