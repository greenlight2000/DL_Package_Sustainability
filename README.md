# README

# Train model

Train LSTM model for OSS sustainability prediction.

1. Go to `train_model` directory
2. Create a conda environment and install required dependencies
    
    ```html
    conda create -n LSTM python=3.8.5
    conda activate LSTM
    pip install -r train_requirement.txt
    ```
    
3. Specify experiment configuration in `LSTM_config.yaml` before running.
4. Train and evaluate LSTM model on dataset at `../mvts_dataset` by running:
    
    ```python
    python LSTM.py LSTM_config.yaml
    ```
    
    The program will automatically train `repeat_times`*(`start_timestep`-`start_timestep`+1)  number of LSTM-based timeseries classification models. And the models and corresponding evaluation results will be generated under `output_path`. (all these parameters are specified in step 3)
    

# Explain model

Explain the prediction behavior of a specified model on specified data samples using SHAP algorithm.

1. Go to shap_explain directory
2. Create a new conda environment and install required dependencies.(note that it should be a separated environment from the model-training environment, due to some SHAP’s compatibility issues)
    
    ```python
    conda create -n SHAP python=3.8.5
    conda activate SHAP
    pip install -r explain_requirement.txt
    ```
    
3. Specify experiment configuration in `shap4LSTM_config.yaml` before running. 
4. Explain LSTM model’s behavior by running:
    
    ```python
    python shap4LSTM.py shap4LSTM_config.yaml
    ```
    
    The program will automatically use the model at `model_path` to inference. And SHAP algorithm will be used to observe the inference results and assign weights to each feature at each timestep of each project. Corresponding explanatory and analysis results (specified by `mode`) will be generated under `output_path`(all these parameters are specified in step 3)