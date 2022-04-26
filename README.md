# Descriptions
Install the dependencies \
`python install --user -r requirements.txt` 

Train the model \
`python model.train.py`

Test the model \
`python model.test.py`

If you want to change the datasets, you can copy your datasets into the `train` folder. The codes will automatically detect the folder listed in the `train` folder as your class label.

# Datasets
You can download the complete dataset <a href="https://github.com/soerenab/AudioMNIST">here</a>. The dataset consists of 30,000 audio samples of spoken digits (0–9) from 60 different speakers.