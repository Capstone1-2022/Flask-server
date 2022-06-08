. ~/.bashrc
module load anaconda3/5.3.1

conda activate PPUU
conda install -n PPUU nb_conda_kernels


cd /scratch/pp1953/cml/ass/a-PyTorch-Tutorial-to-Object-Detection/Object-Detection-Custom-Dataset-pytorch/
python modified_train.py --name="adam_pretrained" --checkpoint >> "output6.txt"