# Topic Modeling

## Dependencies

```bash
conda create -n topic-modeling python=3.10
conda activate topic-modeling
pip install requirements.txt
```

## Build docker

```bash
docker build -t topic-modeling .
```

## Run docker

```bash
docker run -it topic-modeling /bin/bash
```
