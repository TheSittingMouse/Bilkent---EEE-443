#! /usr/bin/env python

import os
import subprocess

with open ('term_proj.sh', 'w') as rsh:
    rsh.write('''\
#! /bin/bash
git clone https://github.com/tobran/DF-GAN.git
pip install -r requirements.txt
cd DF-GAN/code/
bash scripts/calc_fid.sh ./cfg/coco.yml
bash scripts/sample.sh ./cfg/coco.yml

cd ~
git clone https://github.com/davidstap/AttnGAN.git
cd AttnGAN/code/
python main.py --cfg cfg/eval_coco.yml --gpu 0
cd ..
cd eval
cd FID
python fid_score.py --gpu 0 --batch-size 50 --path1 /home/emirhan/T2I_CL/AttnGAN/data/coco/images --path2 /home/emirhan/AttnGAN/models/coco_AttnGAN2/single


cd ~
git clone https://github.com/JuhongPark/DM-GAN.git
cd DM-GAN/code
python main.py --cfg cfg/eval_coco.yml --gpu 0
cd ..
cd eval
cd FID
python fid_score.py --gpu 0 --batch-size 50 --path1 /home/emirhan/DM-GAN/data/coco/images --path2 /home/emirhan/DM-GAN/models/coco_DMGAN/valid/single




cd ~
git clone https://github.com/huiyegit/T2I_CL.git
cd T2I_CL/AttnGAN+CL/code
python main.py --cfg cfg/eval_coco.yml --gpu 0
cd ..
cd eval
cd FID
python fid_score.py --gpu 0 --batch-size 50 --path1 /home/emirhan/T2I_CL/AttnGAN+CL/data/coco/images --path2 /home/emirhan/T2I_CL/AttnGAN+CL/models/netG_epoch_1000/valid/single


cd ~
cd T2I_CL/DM-GAN+CL/code
python main.py --cfg cfg/eval_coco.yml --gpu 0
cd ..
cd eval
cd FID
python fid_score.py --gpu 0 --batch-size 50 --path1 /home/emirhan/T2I_CL/DM-GAN+CL/data/coco/images --path2 /home/emirhan/T2I_CL/DM-GAN+CL/models/netG_epoch_200/valid/single
''')

os.chmod('term_proj.sh', 777)
subprocess.run(['./term_proj.sh'])

