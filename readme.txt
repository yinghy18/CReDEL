First we should train the CReDEL main part ---- the contrastive model. simply run:
        cd ./contrastive_model_training
		python main.py

Then we need to train a entity typing module.
Take the training data is consist of raw_train.txt and tagged_train.txt in ./data/; the UMLS is in ./dict/
The format of tagged file is [{"begin":i, "end":j, "pharse":"the pharse"},...].

The training command is: 
		python main.py --train_batch_size 256 --save_step 1000 --gradient_accumulation_steps 16 --learning_rate 0.00005 --train_epoch 3

Finally we need to refine the DSNER dataset for DSNER methods. Take the BC5CDR dataset annotated by UMLS subset.
Simply:
        run refine_label.sh
		
#pretrained models and data not provided in this source code due to copyright issues, 
#but we provide the data generation method in the paper.

