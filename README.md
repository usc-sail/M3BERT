# M3BERT
A music transformer that extracts representations of audio using several hundreds of thousands of music clips. Fine-tuning is done with diverse end-tasks to enrich the pre-trained representations. More details can be found in the paper "Multi-modal, Multi-task, Music BERT: A Context-Aware Music Encoder Based on Transformers," accessible at https://www.researchgate.net/publication/363811441_Multi-modal_Multi-task_Music_BERT_A_Context-Aware_Music_Encoder_Based_on_Transformers

## Requirements
This package is built in pytorch. If training using a large amount of data, GPU capabilities is recommended. You can install the required packages with

<code>pip install requirements.txt</code>

## Data
Data used to train the M3BERT model can be found at http://millionsongdataset.com/, https://sites. google.com/view/contact4music4all, https://github.com/MTG/mtg-jamendo-dataset, and https://github.com/mdeff/fma.

The datasets for fine-tuning M3BERT can be found at https://github.com/MTG/mtg-jamendo-dataset, http://anasynth.ircam.fr/home/media/ExtendedBallroom/, https://cvml.unige.ch/ databases/DEAM/, https://www.tensorflow.org/datasets/catalog/gtzan, and https://staff.aist.go.jp/m.goto/RWC-MDB/rwc-mdb-i.html.

Any data can be used, so long as you can save the data into numpy files and can point to these npy files in a csv.

## General usage

First, you will save your data into a csv file, where each csv has a column with the filename, length of file (should be less than 30s) and any file-level labels (genre, instrument, etc.). Once this csv is stored and the npy files are generated for your data (refer to paper for our feature extraction), you will want to create a config file that points to this csv. From there, you will be using the runner_m3bert.py file extensively, with different flags for pre-training and for fine-tuning.

## Pre-training and fine-tuning

To pre-train the M3BERT model, you would typically run a command like:
python runner_m3bert.py --train --config config/my_config.yaml --logdir my_logdir

To run the fine-tuning step, you may run something like:
python runner_m3bert.py --train_mtl --config config/my_config.yaml --logdir my_logdir/ --ckpt m3bert-500000.ckpt --ckpdir result/my_ckpt_dir/m3bert/ --frozen --dckpt my_dckpt

Note that many times you can use the same config file for pre-training and for fine-tuning: the config file has separate sections to dictate hyperparameters for each part of the process.

Tensorboard is highly recommended for evaluating training loss. The tensorboard will output masked, reconstructed, and original samples and gives you a good idea of how training loss is developing over time. Correlations between features can be calculated using outputs/corr_analysis.py.
