question=$1
train_file=$2
test_file=$3
output_file=$4

if [[ ${question} == "1" ]]; then
python q1_naivebaeyes.py $train_file $test_file $output_file
fi

if [[ ${question} == "2" ]]; then
python q2_svm.py $train_file $test_file $output_file
fi

if [[ ${question} == "3" ]]; then
python q3_eval_models.py $train_file $test_file $output_file
fi

