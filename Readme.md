## Sequential models 
The base model is CARCA, the SOTA model by 2023. There will be some improvements to get better results based.
This changes can be 

### Data
The data can be found on internet in different forms but they do not match in the number of items or users,
nor in the embedding representation since this one for example comes from images,
and other from the description encoded with Sbert. Please to obtain similar results (taking into account the seed)
use the following dataset.

Preprocessed data: "https://drive.google.com/drive/folders/1a_u52mIEUA-1WrwsNZZa-aoGJcMmVugs?usp=sharing"

### Pytorch environment:
* matplotlib==3.3.4
* numpy==1.21.1
* pandas==1.1.5
* scikit_learn==1.3.0
* tqdm==4.65.0
* transformers==4.18.0
* torch==2.0.0

### To run the respective dataset using, please use the below commands
To run the respective dataset using PyTorch, please use the below commands
- python main.py

The code in main.py has a loop over the datasets names. Please chose those that you want to test and delete the others.
Do not use other names than those in main.py ('Fashion', 'Beauty', 'Men', 'Video_Games')

