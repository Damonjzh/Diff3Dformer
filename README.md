# Diff3Dformer: Leveraging Slice Sequence Diffusion for Enhanced 3D CT Classification with Transformer Networks (MICCAI-2024)

Arxiv: https://arxiv.org/abs/2406.17173 
## Cluster-ViT
The input data of the the cluster-ViT prognosis model follows the format:
```
patient0.npy
|
├── Representation   # (N x M) N is the number of slices for each patient, and M is dimention of representations
├── position # (n x 3) Cordinates for slices in original CT scans
├── cluster # (n x 1) Cluster assignments of representations
├── Dead # (n x 1) 1 means that the patient had common pneumonia for diagnostic task and the patient died within 1 year for prognostic task
```

Run the following steps to train the cluster-ViT prognosis model and obtain the slice-level and patient-level risk scores.
```
cd cluster-ViT
python main.py
    --lr_drop 100
    --epochs 100
    --group_Q
    --batch_size 4
    --dropout 0.1
    --sequence_len 15000
    --weight_decay 0.0001
    --seq_pool
    --dataDir /dataset
    --lr 1e-5
    --mixUp
    --withEmbeddingPreNorm
    --max_num_cluster 64
    --expname test
```

You can used our pretrained model for CCCCII data. The preprocessed CCCCII data and pretrained model can be found at [https://drive.google.com/drive/folders/1eNIEUEAoZH9GIX0ewoBkTOJ2trBGyBmO?usp=sharing](https://drive.google.com/drive/folders/1rUcMb51kNHuVzhdKjQNpmFxbRokukCfq). 
