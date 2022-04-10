## Installation

The code is tested with PyTorch 1.8.0 and CUDA 11.1.

1. Install PyTorch and torchvision
```shell
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html

```
2. Install Detectron2
```shell
python -m pip install -e .

```

## Training
Use script `train_faster_rcnn.py` to train the models. The script expect the following parameters,

* -data_dir -> iSAID dataset path
* -config -> Detectron2 config file listing all model and training related configurations
* -output_dir -> Output directory to save checkpoints and logs
* --resume -> Flag to resume the training from the available latest checkpoints
* --eval_only -> Flag used to perform only the evaluation
* --eval_checkpoints -> Path to the checkpoints to use for the evaluation


The configs for training using SA-AutoAug are available at [here](configs/SA_AutoAug).

## Evaluate pretrained models
Run the following command to evaluate the provided pretrained models,

```shell
python train_faster_rcnn.py -data_dir <path to iSAID dataset> -output_dir <path to output directory to save logs> --eval_only --eval_checkpoints <path to the pretrained model>

```

## Visualization
The visualizations can be generated using the script `visualize_detections.py`.

Should you have any questions, please contact at muhammad.maaz@mbzuai.ac.ae or hanoona.bangalath@mbzuai.ac.ae