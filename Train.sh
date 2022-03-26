echo "GPU ID  " $1

python TrainFair.py --gpu $1 >> alpha1_beta1_t1.txt

python TrainFair.py --gpu $1 --alpha 0 >> alpha0_beta1_t1.txt

python TrainFair.py --gpu $1 --beta 0 >> alpha1_beta0_t1.txt

python TrainFair.py --gpu $1 --t 5 >> alpha1_beta1_t5.txt

python TrainFair.py --gpu $1 --t 10 >> alpha1_beta1_t10.txt


