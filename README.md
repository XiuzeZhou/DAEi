# AEi
Autoencoders for Drug-Target Interaction Prediction

- **Run AEi**:

$ python AEi.py --path datasets/ --data_name Enzyme --epoches 300 --batch_size 256 --hidden_size 512 --reg 0.000001 --keep_rate 1.0 --lr 0.001 --min_loss 0.01 --cv 10 --loss_type square --mode dti

- **Run DAEi**:

$ python DAEi.py --path datasets/ --data_name Enzyme --epoches 300 --batch_size 256 --hidden_size 512 --regs [0.000001,0.000001,0.000001] --noise_level 0.00001 --lr 0.001 --min_loss 0.01 --cv 10 --loss_type square --mode dti


## Parameter description：
- path：Input data path.
- data_name：Name of dataset
- epoches：Number of epoches.
- batch_size：Batch size.
- hidden_size：Hidden layer size, also Embedding size.
- reg: Regularization for L2.
- keep_rate: Keep_rate of dropout.
- lr: Learning rate.
- min_loss: The minimum value for stopping loss function.
- cv: K-fold Cross Validation.
- mode: the mode for training: dti -> drug-target interactions; tdi -> target-drug interactions.
