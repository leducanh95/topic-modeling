# Topic Modeling
Clustering news headlines into groups. Topic modeling and document clustering are techniques used to manipulate and search collections of text for information retrieval and potential knowledge discovery.

## Dataset
### Context
This contains data of news headlines published over a period of nineteen years.

Sourced from the reputable Australian news source ABC (Australian Broadcasting Corporation)

Agency Site: (http://www.abc.net.au)
### Content
Format: CSV ; Single File

1. **publish_date**: Date of publishing for the article in yyyyMMdd format

2. **headline_text**: Text of the headline in Ascii , English , lowercase

Start Date: **2003-02-19** ; End Date: **2021-12-31**

*Source data*: https://www.kaggle.com/datasets/therohk/million-headlines
## Dependencies

```bash
conda create -n topic-modeling python=3.10
conda activate topic-modeling
pip install requirements.txt
```

## Run
Download data from the link above, save date to folder `data/01_input`. Change parameters in `config/config.json`. Run command:
```bash
python main.py
```


<!-- ## Build docker

```bash
docker build -t topic-modeling .
```

## Run docker

```bash
docker run -it topic-modeling /bin/bash
``` -->
