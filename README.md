
[version-image]: https://img.shields.io/badge/python-3.8-brightgreen
[version-url]: https://www.python.org/
# HiUrNet: Explainable Hierarchical Urban Representation Learning for Commuting Flow Prediction

[![Python Version][version-image]][version-url]

![Overview](/assets/outline.png)


This is the implementation of the HiUrNet in the paper: **Explainable Hierarchical Urban Representation Learning for Commuting Flow Prediction** in **ACM SIGSPATIAL 2024**. 
We developed a heterogeneous graph-based model to generate meaningful region embeddings at multiple spatial resolutions for predicting different types of inter-level OD flows.

## Dependencies

* numpy
* pandas
* torch 
* torch_geometric
* matplotlib
* scikit-learn
* jismesh

## Prepare environment
```
conda create -n HiUrNet
conda activate HiUrNet
conda install pip
python -m pip install -r requirement.txt
```

After downloading this repository, run:

```
cd HiUrNet
mkdir result
```

### About Data
- OD data

The OD data used in the paper come from SoftBank Group Corporation, and we are not allowed to open them to the public. 
Currently, we provide a sample file to indicate the format of OD volumes.

The level of the urban unit is indicated by the number of digits for location codes. For example, codes with five digits indicate cities, while codes with nine digits indicate 500 meter mesh girds.

Now we are preparing synthetic data in the same scope using [**PseudoPFlow**](https://onlinelibrary.wiley.com/doi/pdf/10.1111/mice.13285) data. 
Once it is finished, we will update them in this repository.

- Region attributes

You can find them in the `data` folder. You can also download the files from URLs provided in the paper.

- Mesh code

The file demonstrates inclusion relationship between cities and mesh grids, provided by the Japanese government.

- Geographic neighbors

The file includes edge pairs of neighboring mesh grids.


## To Run Codes

### How to generate the prediction

Run the following code:

```
python main.py
```

### How to generate the explanation

Currently, released PyG library does not support explanation for HGT models. 

Please replace the file `torch_geometric/explain/algorithm/utils.py` in the official PyG codes with the file we provide in the `explain` folder. 
Details can be found in this [link](https://github.com/pyg-team/pytorch_geometric/pull/8512). 
The author sincerely appreciates the help from @rachitk

After replacing `util.py`, put the generated `best_model.pth` file into the `explain` folder. 

Then run the explanation generation module:

```
python explain.py
```

You can find the diagram of feature important analysis in the `explain` folder. 
You can also find other explanations in `.csv` files, such as results of edge masks.

### How to run other experiments in the paper

- Different message type combinations: `-flow` `-geo` `-inclusion`
- Different layer number and layer type: `-num_layer` `layer_type`

## Citation
```
@article{cai2024explainable,
  title={Explainable Hierarchical Urban Representation Learning for Commuting Flow Prediction},
  author={Cai, Mingfei and Pang, Yanbo and Sekimoto, Yoshihide},
  journal={arXiv preprint arXiv:2408.14762},
  year={2024}
}
```