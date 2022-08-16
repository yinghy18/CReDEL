Training:
	Take the training data is consist of raw_train.txt and tagged_train.txt in ./data/; the UMLS is in ./dict/
	The format of tagged file is [{"begin":i, "end":j, "pharse":"the pharse"},...].
	The training command is: 
		python main.py --train_batch_size 256 --save_step 1000 --gradient_accumulation_steps 16 --learning_rate 0.00005 --train_epoch 3

Inference:
	If entities extracted by CReDEL is in /mention_type/NCBIparse-sents and ./mention_type/NCBIparse-tags, then the command for entity typing is:
		python predict.py --predict_text_file ./mention_type/NCBIparse-sents --predict_match_file ./mention_type/NCBIparse-tags