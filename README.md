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

# Papers
More details are available in the following papers. Welcome to cite our work if you find it is helpful to your research.

[[Paper](https://doi.org/10.1109/ICASSPW59220.2023.10193003)] 
```
@inproceedings{yuksel23_icassp,
  author       = {Kamer Ali Yuksel and Thiago Castro Ferreira and Ahmet Gunduz and Mohamed Al-Badrashiny and Golara Javadi},
  title        = {A Reference-Less Quality Metric for Automatic Speech Recognition via Contrastive-Learning of a Multi-Language Model with Self-Supervision},
  booktitle    = {IEEE International Conference on Acoustics, Speech, and Signal Processing, ICASSP 2023, Rhodes Island, Greece, June 4-10, 2023},
  pages        = {1--5},
  publisher    = {IEEE},
  year         = {2023},
  url          = {https://doi.org/10.1109/ICASSPW59220.2023.10193003},
  doi          = {10.1109/ICASSPW59220.2023.10193003}
}
```
[[Paper](https://www.isca-speech.org/archive/pdfs/interspeech_2023/yuksel23_interspeech.pdf)]
```
@inproceedings{yuksel23_interspeech,
  author={Kamer Ali Yuksel and Thiago Castro Ferreira and Golara Javadi and Mohamed Al-Badrashiny and Ahmet Gunduz},
  title={{NoRefER: a Referenceless Quality Metric for Automatic Speech Recognition via Semi-Supervised Language Model Fine-Tuning with Contrastive Learning}},
  year=2023,
  booktitle={Proc. INTERSPEECH 2023},
  pages={466--470},
  doi={10.21437/Interspeech.2023-643}
}
```

