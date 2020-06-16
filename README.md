# AEi
Autoencoders for Drug-Target Interaction Prediction

- **Run DAEi**:

$ python DAEi.py --path datasets/ --data_name GPCR --epoches 300 --batch_size 256 --hidden_size 512 --regs [0.000001,0.000001,0.000001] --noise_level 0.00001 --lr 0.001 --min_loss 0.01 --cv 10 --loss_type square --mode dti



