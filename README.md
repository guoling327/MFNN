# MFNN

Welcome to the source code repository for our paper: **Beyond Pairwise Dependence: A Multi-Filter Fusion Network for Graph Representation Learning**


# Prerequisites:
Ensure you have the following libraries installed:
```
pytorch
pytorch-geometric
networkx
numpy
```

# Exp1: LearningFilters
go to folder `./LearningFilters`

## Run experiment :
```sh
cd LearningFilters
python training.py
```

# Exp2: Node Classification 
go to folder `./node_classify/src`

## Run experiment with Cora:

```sh
cd node_classify/src
python run_node_exp.py --RPMAX 100 \
        --net MFNN\
        --dataset cora \
        --lr 0.06 \
        --hidden 28 \
         --Order 2 \
        --alpha 0.5 \
        --weight_decay 5e-3 \
        --dropout 0.6
```

Before the training commences, the script will download and preprocess the respective graph datasets. 
Subsequently, it performs the appropriate graph-lifting procedure (this process might a while).


# Exp3: graph classification
go to folder `./graph_classify`

We prepared individual scripts for each experiment. The results are written in the
`exp/results/` directory and are also displayed in the terminal once the training is
complete. 
```sh
cd graph_classify
sh scripts/MFNN-PROETINS.sh
```


 
Thank you for your interest in our work. If you have any questions or encounter any issues while using our code, please don't hesitate to raise an issue or reach out to us directly.


