# NoRefER
This is a repo for reproducing the results presented at NoRefER paper.

- **Model checkpoints** are available in the link below:

  https://drive.google.com/file/d/1KgMiU_9asfDEKLTkc8sqagG1fIk84nV_/view?usp=sharing

- **test datasets** are uploaded as csv files. These files are used for reported results in the paper.

- **The code for the results of NoRefER-Self and NoeRefER-Semi** is provided in main_noref.py. To run the code, please especify the model and the filename in your command.

  <code>python main_noref.py --filename filename.csv --modelname model_checkpoint.ckpt</code>

  example:

  <code>python main_noref.py --filename en-common.csv --modelname self_super.ckpt</code>

- **The code for the baseline results** and calculating perplexity is also provided in perplexity_noref.py. You just need to specify the filename to run this code. 

  example:

  <code>python perplexity_noref.py --filename en-common.csv</code>
