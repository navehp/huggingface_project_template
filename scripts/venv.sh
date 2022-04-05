export PROJECT_NAME=template
export DIR_NAME=project_template

### Create Dir ###
mkdir ${DIR_NAME}

### Install Anaconda ###
mkdir ~/tmp
cd ~/tmp
wget https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh
bash Anaconda3-2020.02-Linux-x86_64.sh
source ~/anaconda3/etc/profile.d/conda.sh
cd ../
rm -r tmp

### Create Venv ###
echo "Conda Pytorch Env Setup"
conda create --name ${PROJECT_NAME} python==3.9.7  # Run this by hand before the other commands

conda activate ${PROJECT_NAME}
conda install pip
pip install numpy scipy sklearn
pip install torch==1.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html  # make sure to install the right torch
pip install transformers datasets
#pip install -r requirements.txt

### Setup Wandb ###
pip install wandb
wandb login  # Only do this once on each server
# paste wandb api code:

### Clone git repo ###
#echo "Clone git repository"
#mkdir -p ~/dev
#cd ~/dev/
#git clone <link to git repo>
#echo "export GIT_REPO=\$HOME/dev/<repo name>/" >> ~/.bash_profile

### Define Eliases ###
conda deactivate
echo "alias 'conda_env'='source \$HOME/anaconda3/etc/profile.d/conda.sh'" >> ~/.bash_profile
echo "alias '${PROJECT_NAME}'='conda_env && conda activate ${PROJECT_NAME} && export PYTHONPATH=\$HOME/${DIR_NAME}/:\$PYTHONPATH && cd \$HOME/${DIR_NAME}/'" >> ~/.bash_profile
source ~/.bash_profile
