All the notebooks are real-world notebooks randomly picked from Kaggle (subject to the criteria mentioned in Section 6.1). We have placed them in directories that follow the scheme `<user>/<notebook id>`, which one can use to find the original Kaggle notebook. For example, `lextoumbourou/feedback3-eda-hf-custom-trainer-sift` originates from https://www.kaggle.com/code/lextoumbourou/feedback3-eda-hf-custom-trainer-sift/.

The datasets we use are also the original datasets available in Kaggle. They are not in this repository though. See [Download the datasets](https://github.com/ADAPT-uiuc/dias-benchmarks/blob/main/REPRODUCE.md#download-the-datasets).

The notebooks have been slightly modified. We give further details in the paper regarding the reasons for modifying the notebooks. Here, we provide a summary. We have disabled:
- Networking or shell code i.e., cells like `pip install` have been disabled (out of scope and variant).
- Plotting and machine-learning code (out of scope)
- IPython magic functions (implementation limitation; see Section 5.1)

Also, in some cases we have replicated the dataset to avoid measurement noise.