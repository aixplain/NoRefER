# XAI 
Here we use NORefER as a tool for XAI using token attentions of this model.

- **Model checkpoints** are available in the link below:

  https://drive.google.com/file/d/1KgMiU_9asfDEKLTkc8sqagG1fIk84nV_/view?usp=sharing

In these experiments we used **NoeRefER-Semi** model parameters.

- **Test datasets** are uploaded as [csv files](https://github.com/aixplain/NoRefER/tree/main/dataset). These files are used for reported results in the paper. Bellow you can find the link to download "LibriSpeech" and "CommonVoice" datasets.

  Link to download **LibriSpeech**: https://huggingface.co/datasets/librispeech_asr
  
  Link to download **CommonVoice**: https://huggingface.co/mozilla-foundation

- To reproduce the result for baseline run the following notebooks:

  [CTC](https://github.com/aixplain/NoRefER/blob/main/icassp-xai/baseline/CTC/ASR_confidence_estimation.ipynb)

  [Whisper](https://github.com/aixplain/NoRefER/blob/main/icassp-xai/baseline/ASR_whisper/whisper_base.ipynb)

[[Paper](https://arxiv.org/pdf/2401.11268.pdf)]
```
@article{javadi2024word,
  title={Word-Level ASR Quality Estimation for Efficient Corpus Sampling and Post-Editing through Analyzing Attentions of a Reference-Free Metric},
  author={Javadi, Golara and Yuksel, Kamer Ali and Kim, Yunsu and Ferreira, Thiago Castro and Al-Badrashiny, Mohamed},
  journal={arXiv preprint arXiv:2401.11268},
  year={2024}
}
```


