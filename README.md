# Learn and Compose Fluid Component behavior using DeCNN and GNN
DeCNN, GNN, and pipe flow simulation code associated with our accepted IDETC 2020 paper: "Learning to Abstract and Compose Mechanical Device Function and Behavior."

![Alt text](/overview.jpg)

## License
This code is licensed under the MIT license. Feel free to use all or portions for your research or related projects so long as you provide the following citation information:

Jun Wang, Kevin Chiu, and Mark Fuge. "Learning to Abstract and Compose Mechanical Device Function and Behavior." IDETC (2020).
https://doi.org/10.1115/DETC2020-22714

    @inproceedings{wang2020learning,
          title={Learning to Abstract and Compose Mechanical Device Function and Behavior},
          author={Wang, Jun and Chiu, Kevin and Fuge, Mark},
          booktitle={International Design Engineering Technical Conferences and Computers and Information in Engineering Conference},
          volume={84003},
          pages={V11AT11A009},
          year={2020},
          organization={American Society of Mechanical Engineers}
     }

## Required packages

### Network training
- tensorflow<2.0.0
- h5py
- graph_nets
- numpy
- matplotlib
- networkx
- sonnet

### Simulation
- [FEniCS project](https://fenicsproject.org/download/)

## Usage

### Train/evaluate DeCNN

```bash
python main_new.py startover ae_new
```

positional arguments:
    
```
mode	startover or evaluate
```

optional arguments:

```
--batch_size      training batch size
--save_interval   number of intervals for saving the trained model and plotting results
--train_steps     number of training steps
```

### Train/evaluate GNN

```bash
python main_tensor.py train gnn_tensor
```

positional arguments:
    
```
mode	train or evaluate
```

optional arguments:

```
--batch_size      training batch size
--save_interval   number of intervals for saving the trained model and plotting results
--train_steps     number of training steps
```

## Dataset

We have two types of basic pipes

<!-- ![Alt text](/pipes_a.jpg){:height="50%" width="50%"}  -->
<img src="/pipes_a.jpg" width="50%" height="50%"/>

and five unique combinations of two-pipe composition

<!-- ![Alt text](/combinations.jpg){:height="50%" width="50%"} -->
<img src="/combinations.jpg" width="50%" height="50%"/>

### DeCNN database
Because the data size is large, please find the dataset [here](https://drive.google.com/drive/folders/1yE4b3Vmk74sf5Lo8jqGksF8fpKyzFVlZ?usp=sharing).

### GNN database
Please find the dataset [here](https://drive.google.com/drive/folders/1rfy6kZEKiXCrF-UiUQT0pzbScOVmbidB?usp=sharing).

## Results

### Performance of DeCNN for learn-to-abstract:

<img src="/visual_str_decnn_dis_per.png" width="30%" height="30%"/> &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; <img src="/visual_L_decnn_dis_per.png" width="30%" height="30%"/>

### Performance of GNN for learn-to-compose:

<img src="/visual_SS_gnn_fig13.png" width="30%" height="30%"/> &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; <img src="/visual_SL_gnn_fig13.png" width="30%" height="30%"/>

<img src="/visual_LStr_gnn_fig13.png" width="30%" height="30%"/> &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; <img src="/visual_LL_gnn_fig13.png" width="30%" height="30%"/>
