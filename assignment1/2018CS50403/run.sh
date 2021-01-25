data_dir=$1
out_dir=$2
question=$3
part=$4
if [[ ${question}_${part} == "1_a" ]]; then
python q1_linear.py $data_dir $out_dir a
fi

if [[ ${question}_${part} == "1_b" ]]; then
python q1_linear.py $data_dir $out_dir b
fi

if [[ ${question}_${part} == "1_c" ]]; then
python q1_linear.py $data_dir $out_dir c
fi

if [[ ${question}_${part} == "1_d" ]]; then
python q1_linear.py $data_dir $out_dir d
fi

if [[ ${question}_${part} == "1_e" ]]; then
python q1_linear.py $data_dir $out_dir e
fi

if [[ ${question}_${part} == "2_a" ]]; then
python q2_stochastic.py $data_dir $out_dir
fi

if [[ ${question}_${part} == "2_b" ]]; then
python q2_stochastic.py $data_dir $out_dir
fi

if [[ ${question}_${part} == "2_c" ]]; then
python q2_stochastic.py $data_dir $out_dir
fi

if [[ ${question}_${part} == "2_d" ]]; then
python q2_stochastic.py $data_dir $out_dir
fi


if [[ ${question}_${part} == "3_a" ]]; then
python q3_logistic_regression.py $data_dir $out_dir a
fi

if [[ ${question}_${part} == "3_b" ]]; then
python q3_logistic_regression.py $data_dir $out_dir b
fi

if [[ ${question}_${part} == "4_a" ]]; then
python q4_GDA.py $data_dir $out_dir a
fi

if [[ ${question}_${part} == "4_b" ]]; then
python q4_GDA.py $data_dir $out_dir b
fi

if [[ ${question}_${part} == "4_c" ]]; then
python q4_GDA.py $data_dir $out_dir c
fi

if [[ ${question}_${part} == "4_d" ]]; then
python q4_GDA.py $data_dir $out_dir d
fi

if [[ ${question}_${part} == "4_e" ]]; then
python q4_GDA.py $data_dir $out_dir e
fi