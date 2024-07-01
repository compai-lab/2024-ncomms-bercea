<p align="center">
<img src="https://github.com/ci-ber/GenAI_UAD/assets/106509806/a1b6785e-0844-422b-911e-69da4e7aced1" width=200>
</p>

<h1 align="center">
  <br>
Evaluating Normative Learning in Generative AI for Robust Medical Anomaly Detection
  <br>
</h1>
</h1>
  <p align="center">
    <a href="https://ci.bercea.net">Cosmin Bercea</a> •
    <a href="https://www.neurokopfzentrum.med.tum.de/neuroradiologie/mitarbeiter-profil-wiestler.html">Benedikt Wiestler</a> •
    <a href="https://aim-lab.io/author/daniel-ruckert/">Daniel Rueckert </a> •
    <a href="https://compai-lab.github.io/author/julia-a.-schnabel/">Julia A. Schnabel </a>
  </p>
<h4 align="center">Official repository of the paper</h4>
<h4 align="center">Under Review</h4>
<h4 align="center"> <a href="https://openreview.net/pdf?id=kTpafpXrqa](https://www.researchsquare.com/article/rs-3749187/v1)">Paper</a> </h4>

<p align="center">
<img src="https://github.com/ci-ber/GenAI_UAD/assets/106509806/5303973d-6bde-4d20-bd44-c4ff84deb9cd">
</p>


## Citation

If you find our work useful, please cite our paper:
```
@article{bercea2023evaluating,
  title={Evaluating Normative Learning in Generative AI for Robust Medical Anomaly Detection},
  author={Bercea, Cosmin and Wiestler, Benedikt and Rueckert, Daniel and Schnabel, Julia},
  year={2023}
}
```

> **Abstract:** *In Generative Artificial Intelligence (AI) for medical imaging, normative learning involves training AI models on large datasets of typical images from healthy volunteers, such as MRIs or CT scans. These models acquire the distribution of normal anatomical structures, allowing them to effectively detect and correct anomalies in new, unseen pathological data. This approach allows the detection of unknown pathologies without the need for expert labeling.
Traditional anomaly detection methods often evaluate the anomaly detection performance, overlooking the crucial role of normative learning. In our analysis, we introduce novel metrics, specifically designed to evaluate this facet in AI models. We apply these metrics across various generative AI frameworks, including advanced diffusion models, and rigorously test them against complex and diverse brain pathologies. Our analysis demonstrates that models proficient in normative learning exhibit exceptional versatility, adeptly detecting a wide range of unseen medical conditions.* 

## Setup and Run

The code is based on the deep learning framework from the Institute of Machine Learning in Biomedical Imaging: https://github.com/compai-lab/iml-dl

### Framework Overview 

<p align="center">
<img src="https://github.com/ci-ber/autoDDPM/assets/106509806/678c5d6c-efb0-4934-a635-284b06636a78">
</p>

#### 1). Set up wandb (https://docs.wandb.ai/quickstart)

Sign up for a free account and login to your wandb account.
```bash
wandb login
```
Paste the API key from https://wandb.ai/authorize when prompted.

#### 2). Clone repository

```bash
git clone https://github.com/ci-ber/GenAI_UAD.git
cd GenAI_UAD
```

#### 3). Install PyTorch 

*Optional* create virtual env:
```bash
conda create --name genai python=3.8.0
conda activate genai
```

> Example installation: 
* *with cuda*: 
```
pip3 install torch==1.9.1+cu111 torchvision==0.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
```
* *w/o cuda*:
```
pip3 install torch==1.9.1 torchvision==0.10.1 -f https://download.pytorch.org/whl/torch_stable.html
```

#### 4). Install requirements

```bash
pip install -r pip_requirements.txt
```

#### 5). Download datasets 

<h4 align="center"><a href="https://brain-development.org/ixi-dataset/">IXI</a> • <a href="https://fastmri.org">FastMRI</a> • <a href="https://github.com/microsoft/fastmri-plus"> Labels for FastMRI</a> • <a href="https://fcon_1000.projects.nitrc.org/indi/retro/atlas.html">Atlas (Stroke) </a> </h4>

> Move the datasets to the target locations. You can find detailed information about the expected files and locations in the corresponding *.csv files under data/$DATASET/splits.

> *Alternatively train from scratch on other anatomies and modalities.*

#### 6). Run the pipeline

Run the main script with the corresponding config like this:

```bash
python core/Main.py --config_path ./projects/24_normative_eval/autoddpm.yaml
```
Refer to the autoddpm.yaml (or *other_metho*.yaml) for the default configuration. 

# That's it, enjoy! :rocket:
