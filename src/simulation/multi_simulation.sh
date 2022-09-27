echo "parallel"
python create_simulation_data.py --speed 10 20 30 40 50 --size 10 20 30 40 50 --path -500 0 500 0 --number-of-runs 6 --output-dir ../../data/simulation_data/parallel/
python combine_simulation_data.py --input-dir ../../data/simulation_data/parallel/

echo "offset"
python create_simulation_data.py --speed 10 20 30 40 50 --size 10 20 30 40 50 --path -500 0 500 75 --number-of-runs 6 --output-dir ../../data/simulation_data/offset/
python combine_simulation_data.py --input-dir ../../data/simulation_data/offset/

echo "offset inverse"
python create_simulation_data.py --speed 10 20 30 40 50 --size 10 20 30 40 50 --path -500 75 500 0 --number-of-runs 6 --output-dir ../../data/simulation_data/offset_inverse/
python combine_simulation_data.py --input-dir ../../data/simulation_data/offset_inverse/

echo "orthogonal"
python create_simulation_data.py --speed 10 20 30 40 50 --size 10 20 30 40 50 --path 0 0 0 75 --number-of-runs 6 --output-dir ../../data/simulation_data/orthogonal/
python combine_simulation_data.py --input-dir ../../data/simulation_data/orthogonal/

echo "far off parallel"
python create_simulation_data.py --speed 10 20 30 40 50 --size 10 20 30 40 50 --path -500 150 500 150 --number-of-runs 6 --output-dir ../../data/simulation_data/far_off_parallel/
python combine_simulation_data.py --input-dir ../../data/simulation_data/far_off_parallel/

echo "mult_path"
python create_simulation_data.py --speed 10 20 30 40 50 --size 10 20 30 40 50 --path -500 0 -300 75 -100 0 100 75 300 0 500 75  --number-of-runs 6 --output-dir ../../data/simulation_data/mult_path/
python combine_simulation_data.py --input-dir ../../data/simulation_data/mult_path/
