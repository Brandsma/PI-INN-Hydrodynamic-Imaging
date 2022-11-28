echo "parallel"
python create_simulation_data.py --speed 10 20 30 40 50 --size 10 20 30 40 50 --path -500 0 500 0 --number-of-runs 32 --output-dir ../../../data/simulation_data/parallel/
python combine_simulation_data.py --input-dir ../../../data/simulation_data/parallel/

echo "offset"
python create_simulation_data.py --speed 10 20 30 40 50 --size 10 20 30 40 50 --path -500 0 500 75 --number-of-runs 32 --output-dir ../../../data/simulation_data/offset/
python combine_simulation_data.py --input-dir ../../../data/simulation_data/offset/

echo "offset inverse"
python create_simulation_data.py --speed 10 20 30 40 50 --size 10 20 30 40 50 --path -500 75 500 0 --number-of-runs 32 --output-dir ../../../data/simulation_data/offset_inverse/
python combine_simulation_data.py --input-dir ../../../data/simulation_data/offset_inverse/

echo "orthogonal"
python create_simulation_data.py --speed 10 20 30 40 50 --size 10 20 30 40 50 --path 0 0 0 75 --number-of-runs 32 --output-dir ../../../data/simulation_data/orthogonal/
python combine_simulation_data.py --input-dir ../../../data/simulation_data/orthogonal/

echo "far off parallel"
python create_simulation_data.py --speed 10 20 30 40 50 --size 10 20 30 40 50 --path -500 150 500 150 --number-of-runs 32 --output-dir ../../../data/simulation_data/far_off_parallel/
python combine_simulation_data.py --input-dir ../../../data/simulation_data/far_off_parallel/

echo "mult_path"
python create_simulation_data.py --speed 10 20 30 40 50 --size 10 20 30 40 50 --path -500 0 -300 75 -100 0 100 75 300 0 500 75  --number-of-runs 32 --output-dir ../../../data/simulation_data/mult_path/
python combine_simulation_data.py --input-dir ../../../data/simulation_data/mult_path/


echo " ==== Combining simulation data"
simulation_types=("parallel" "offset" "offset_inverse" "orthogonal" "far_off_parallel" "mult_path")
# simulation_types=("offset_inverse")
for simulation_type in ${simulation_types[@]}
do
    echo "Copying ${simulation_type}..."
    combined_simulation_data_files=`ls ../../../data/simulation_data/${simulation_type}/combined*.npy`
    for eachfile in $combined_simulation_data_files
    do
        base=$(basename $eachfile)
        if [ "$base" == "combined.npy" ]; then
            cp $eachfile "../../../data/simulation_data/combined_groups/${simulation_type}_combined_data.npy"
        else
            cp $eachfile "../../../data/simulation_data/combined_groups/${simulation_type}_${base}"
        fi
    done
done

python combine_simulation_data.py --input-dir ../../../data/simulation_data/combined_groups/ --combined

# cd ..

# echo " ==== Training LSTM"
# python train_lstm.py

# echo " ==== Getting Volume"
# python get_volume.py
