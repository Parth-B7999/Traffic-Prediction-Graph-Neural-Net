# Traffic-Prediction-Graph-Neural-Net

Vehicle Traffic Prediction with Spatio-Temporal Graph Convolution Neural Networks (T-GCN) | Self Project          
•	Fetched for METR-LA (Los Angeles Metropolitan Traffic) dataset in Pytorch Geometric Temporal Data sequential loader                               
•	Developed T-GCN Neural Net model; uses Spatial and Temporal aggregation as Graph Convolution (GNN) & GRU respectively

## Dataset
Traffic forecasting dataset based on Los Angeles Metropolitan traffic                                                                         
207 loop detectors on highways                                                                                        
Time Rnage: March 2012 - June 2012                                                                          
From the paper: Diffusion Convolutional Recurrent Neural Network                                                                               

## Data Sample : Static Graph and dynamic features
207 nodes in each graphs                                                                                                                                                
12 timesteps per sequence (12 x 5 min = 60 min)                                                                             
INPUTS : 2 features per node (speed, time)                                                           
OUTPUTS : Labels for 12 future timesteps (normalized speed) --> node regression                                                                                        
Edge_attr is build based on the distances between sensors + threshold                                                                                   
Further details: https://pytorch-geometric-temporal.readthedocs.io/en/latest/_modules/torch_geometric_temporal/dataset/metr_la.html#METRLADatasetLoader   
Raw data: https://graphmining.ai/temporal_datasets/METR-LA.zip                       

## Model
A3TGCN is an extension of TGCN that uses attention
The spatial aggregation uses GCN, the temporal aggregation a GRU
We can pass in periods to get an embedding for several timesteps
This embedding can be used to predict several steps into the future = output dimension
Source Code: https://pytorch-geometric-temporal.readthedocs.io/en/latest/_modules/torch_geometric_temporal/nn/recurrent/attentiontemporalgcn.html#A3TGCN
