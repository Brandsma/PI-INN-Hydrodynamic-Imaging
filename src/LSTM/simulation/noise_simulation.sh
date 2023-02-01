echo "low_noise_parallel"
python create_simulation_data.py --speed 10 20 30 40 50 --size 10 20 30 40 50 --path -500 0 500 0 --number-of-runs 32 --output-dir ../../../data/simulation_data/noise/low_noise_parallel/ --noise-power 2.5e-6
python combine_simulation_data.py --input-dir ../../../data/simulation_data/noise/low_noise_parallel/

# echo "medium_noise_parallel"
# python create_simulation_data.py --speed 10 20 30 40 50 --size 10 20 30 40 50 --path -500 0 500 0 --number-of-runs 32 --output-dir ../../../data/simulation_data/noise/medium_noise_parallel/
# python combine_simulation_data.py --input-dir ../../../data/simulation_data/noise/medium_noise_parallel/

echo "high_noise_parallel"
python create_simulation_data.py --speed 10 20 30 40 50 --size 10 20 30 40 50 --path -500 0 500 0 --number-of-runs 32 --output-dir ../../../data/simulation_data/noise/high_noise_parallel/ --noise-power 2e-4
python combine_simulation_data.py --input-dir ../../../data/simulation_data/noise/high_noise_parallel/

echo "low_noise_saw"
python create_simulation_data.py --speed 10 20 30 40 50 --size 10 20 30 40 50 --path -500 0 -300 75 -100 0 100 75 300 0 500 75  --number-of-runs 32 --output-dir ../../../data/simulation_data/noise/low_noise_saw/  --noise-power 2.5e-6
python combine_simulation_data.py --input-dir ../../../data/simulation_data/noise/low_noise_saw/

# echo "medium_noise_saw"
# python create_simulation_data.py --speed 10 20 30 40 50 --size 10 20 30 40 50 --path -500 0 -300 75 -100 0 100 75 300 0 500 75  --number-of-runs 32 --output-dir ../../../data/simulation_data/noise/medium_noise_saw/
# python combine_simulation_data.py --input-dir ../../../data/simulation_data/noise/medium_noise_saw/

echo "high_noise_saw"
python create_simulation_data.py --speed 10 20 30 40 50 --size 10 20 30 40 50 --path -500 0 -300 75 -100 0 100 75 300 0 500 75  --number-of-runs 32 --output-dir ../../../data/simulation_data/noise/high_noise_saw/ --noise-power 2e-4
python combine_simulation_data.py --input-dir ../../../data/simulation_data/noise/high_noise_saw/


echo " ==== Combining simulation data"
simulation_types=("low_noise_parallel medium_noise_parallel high_noise_parallel low_noise_saw medium_noise_saw high_noise_saw")
for simulation_type in ${simulation_types[@]}
do
    echo "Copying ${simulation_type}..."
    combined_simulation_data_files=`ls ../../../data/simulation_data/noise/${simulation_type}/combined*.npy`
    for eachfile in $combined_simulation_data_files
    do
        base=$(basename $eachfile)
        if [ "$base" == "combined.npy" ]; then
            cp $eachfile "../../../data/simulation_data/noise/combined_groups/${simulation_type}_combined_data.npy"
        else
            cp $eachfile "../../../data/simulation_data/noise/combined_groups/${simulation_type}_${base}"
        fi
    done
done

python combine_simulation_data.py --input-dir ../../../data/simulation_data/noise/combined_groups/ --combined

# cd ..

# echo " ==== Training LSTM"
# python train_lstm.py

# echo " ==== Getting Volume"
# python get_volume.py
