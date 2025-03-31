# Python example code for the George B. Moody PhysioNet Challenge 2025

## What's in this repository?

This repository contains a simple example that illustrates how to format a Python entry for the [George B. Moody PhysioNet Challenge 2025](https://physionetchallenges.org/2025/). If you are participating in the 2025 Challenge, then we recommend using this repository as a template for your entry. You can remove some of the code, reuse other code, and add new code to create your entry. You do not need to use the models, features, and/or libraries in this example for your entry. We encourage a diversity of approaches to the Challenges.

For this example, we implemented a random forest model with several simple features. (This simple example is **not** designed to perform well, so you should **not** use it as a baseline for your approach's performance.) You can try it by running the following commands on the Challenge training set. If you are using a relatively recent personal computer, then you should be able to run these commands from start to finish on a small subset (1000 records) of the training data in a few minutes or less.

## Quick Start with Docker Compose

Here's how to quickly get started with this project using **Docker Compose** to ensure code runs in an isolated environment with all dependencies. We highly recommend using this method for running this project:

1. Make sure you have [Docker](https://docs.docker.com/get-docker/) and [Docker Compose](https://docs.docker.com/compose/install/) installed.


2. Download the training data and place it in the `databases` directory, unzip the corresponding compressed package, and ensure that the training data is in hdf5 format:

        user@computer:~$ mkdir -p databases model predictions

3. Build and start the Docker container:

        user@computer:~$ docker compose up -d

4. This will start a Docker container, enter the running container:

        user@computer:~$ docker exec -it physionet-challenge-2025 bash

    Execute the following command in the container to enter the code storage path `/challenge`

        root@[...]:~$ cd /challenge

5. Run the preprocessing scripts inside the container to generate processed data (results will be saved in the `processed_data` directory). Next, we will take `exams_part0.hdf5` as an example to demonstrate:

        root@[...]:/challenge# python prepare_code15_data.py -i databases/exams_part0.hdf5 -d databases/exams.csv -l databases/code15_chagas_labels.csv -o processed_data/code15
    Note: If you are using the `PTB-XL` or `SaMi-Trop` dataset, please change the ending `code15` to `PTB-XL` or `SaMi-Trop`. For details, please refer to [How do I create data for these scripts?](#how-do-i-create-data-for-these-scripts).

6. Train the model and make predictions:

    First, run the following command to train the model:

        root@[...]:/challenge# python train_model.py -d processed_data -m model -v

    Here, `-d processed_data` specifies the input folder with processed data, `-m model` specifies the output folder for the model, and `-v` enables verbose mode.

    Next, run the following command to use the trained model to make predictions:

        root@[...]:/challenge# python run_model.py -d processed_data -m model -o predictions -v

    Here, `-d processed_data` specifies the input folder with processed data, `-m model` specifies the input folder for the model, `-o predictions` specifies the output folder for predictions, and `-v` enables verbose mode.

    Finally, run the following command to evaluate the model's performance:

        root@[...]:/challenge# python evaluate_model.py -d processed_data -o predictions -s scores.csv

    Here, `-d processed_data` specifies the input folder with processed data, `-o predictions` specifies the input folder with predictions, and `-s scores.csv` specifies the output file for scores.

7. Exit the container when finished:

        root@[...]:/challenge# exit

## How do I run these scripts manually?

First, you can download and create data for these scripts by following the [instructions](#how-do-i-create-data-for-these-scripts) in the following section.

Second, you can install the dependencies for these scripts by creating a Docker image (see below) or [virtual environment](https://docs.python.org/3/library/venv.html) and running

    pip install -r requirements.txt

Third, you need to preprocess your data. For example, if you are using `CODE-15%` dataset:

    python prepare_code15_data.py -i databases/exams_part0.hdf5 -d databases/exams.csv -l databases/code15_chagas_labels.csv -o processed_data/code15

If you are using the `PTB-XL` or `SaMi-Trop` dataset, please change the script name and parameters accordingly. See [How do I create data for these scripts?](#how-do-i-create-data-for-these-scripts) for details.

You can train your model by running

    python train_model.py -d processed_data -m model -v

where

- `processed_data` (input; required) is a folder with the processed data files, which must include the labels; and
- `model` (output; required) is a folder for saving your model.

You can run your trained model by running

    python run_model.py -d processed_data -m model -o predictions

where

- `processed_data` (input; required) is a folder with the processed data files, which will not necessarily include the labels;
- `model` (input; required) is a folder for loading your model; and
- `predictions` (output; required) is a folder for saving your model outputs.

The [Challenge website](https://physionetchallenges.org/2025/#data) provides a training database with a description of the contents and structure of the data files.

You can evaluate your model by pulling or downloading the [evaluation code](https://github.com/physionetchallenges/evaluation-2025) and running

    python evaluate_model.py -d processed_data -o predictions -s scores.csv

where

- `processed_data`(input; required) is a folder with labels for the processed data files, which must include the labels;
- `predictions` (input; required) is a folder containing files with your model's outputs for the data; and
- `scores.csv` (output; optional) is file with a collection of scores for your model.

You can use the provided training set for the `processed_data` files, but we will use different datasets for the validation and test sets, and we will not provide the labels to your code.

## How do I create data for these scripts?

You can use the scripts in this repository to convert the [CODE-15% dataset](https://zenodo.org/records/4916206), the [SaMi-Trop dataset](https://zenodo.org/records/4905618), and the [PTB-XL dataset](https://physionet.org/content/ptb-xl/) to [WFDB](https://wfdb.io/) format. All raw data should be placed in the `databases` directory, and the processed data will be generated in the `processed_data` directory.

Please see the [data](https://physionetchallenges.org/2025/#data) section of the website for more information about the Challenge data.

#### CODE-15% dataset

1. Download and unzip one or more of the `exam_part` files and the `exams.csv` file from the [CODE-15% dataset](https://zenodo.org/records/4916206) and place them in the `databases` directory.

2. Download and unzip the Chagas labels, i.e., the [`code15_chagas_labels.csv`](https://physionetchallenges.org/2025/data/code15_chagas_labels.zip) file and place it in the `databases` directory.

3. Convert the CODE-15% dataset to WFDB format, with the available demographics information and Chagas labels in the WFDB header file, by running

        python prepare_code15_data.py \
            -i databases/exams_part0.hdf5 databases/exams_part1.hdf5 \
            -d databases/exams.csv \
            -l databases/code15_chagas_labels.csv \
            -o processed_data/code15_part0 processed_data/code15_part1

Each `exam_part` file in the [CODE-15% dataset](https://zenodo.org/records/4916206) contains approximately 20,000 ECG recordings. You can include more or fewer of these files to increase or decrease the number of ECG recordings, respectively. You may want to start with fewer ECG recordings to debug your code.

#### SaMi-Trop dataset

1. Download and unzip `exams.zip` file and the `exams.csv` file from the [SaMi-Trop dataset](https://zenodo.org/records/4905618) and place them in the `databases` directory.

2. Download and unzip the Chagas labels, i.e., the [`samitrop_chagas_labels.csv`](https://physionetchallenges.org/2025/data/samitrop_chagas_labels.zip) file and place it in the `databases` directory.

3. Convert the SaMi-Trop dataset to WFDB format, with the available demographics information and Chagas labels in the WFDB header file, by running

        python prepare_samitrop_data.py \
            -i databases/exams.hdf5 \
            -d databases/exams.csv \
            -l databases/samitrop_chagas_labels.csv \
            -o processed_data/samitrop

#### PTB-XL dataset

We are using the `records500` folder, which has a 500Hz sampling frequency, but you can also try the `records100` folder, which has a 100Hz sampling frequency.

1. Download and, if necessary, unzip the [PTB-XL dataset](https://physionet.org/content/ptb-xl/) and place the files in the `databases/ptbxl` directory.

2. Update the WFDB files with the available demographics information and Chagas labels by running

        python prepare_ptbxl_data.py \
            -i databases/ptbxl/records500/ \
            -d databases/ptbxl/ptbxl_database.csv \
            -o processed_data/ptbxl

## Which scripts I can edit?

Please edit the following script to add your code:

* `team_code.py` is a script with functions for training and running your trained model.

Please do **not** edit the following scripts. We will use the unedited versions of these scripts when running your code:

* `train_model.py` is a script for training your model.
* `run_model.py` is a script for running your trained model.
* `helper_code.py` is a script with helper functions that we used for our code. You are welcome to use them in your code.

These scripts must remain in the root path of your repository, but you can put other scripts and other files elsewhere in your repository.

## How do I train, save, load, and run my model?

To train and save your model, please edit the `train_model` function in the `team_code.py` script. Please do not edit the input or output arguments of this function.

To load and run your trained model, please edit the `load_model` and `run_model` functions in the `team_code.py` script. Please do not edit the input or output arguments of these functions.

## How do I run these scripts in Docker?

**We strongly recommend using Docker** to run your code, ensuring it can be reliably executed in any environment. Docker and similar platforms allow you to containerize and package your code with specific dependencies so that your code can be reliably run in other computational environments.

We provide a `docker-compose.yml` file to simplify using Docker. Please refer to the "Quick Start with Docker Compose" section above to get started quickly.

If you want to manually build and run the Docker image, follow these steps:

1. Create the necessary directories:

        user@computer:~$ mkdir -p databases processed_data model predictions

2. Download and place the training data in the `databases` directory.

3. Build the Docker image:

        user@computer:~$ docker build -t physionet-challenge-2025 .

4. Run the Docker container:

        user@computer:~$ docker run -it \
            -v $(pwd)/databases:/challenge/databases \
            -v $(pwd)/processed_data:/challenge/processed_data \
            -v $(pwd)/model:/challenge/model \
            -v $(pwd)/predictions:/challenge/predictions \
            physionet-challenge-2025 bash

5. Execute the code inside the container:

        root@[...]:/challenge# python prepare_code15_data.py -i databases/exams_part0.hdf5 -d databases/exams.csv -l databases/code15_chagas_labels.csv -o processed_data/code15
        root@[...]:/challenge# python train_model.py -d processed_data -m model -v
        root@[...]:/challenge# python run_model.py -d processed_data -m model -o predictions -v
        root@[...]:/challenge# python evaluate_model.py -d processed_data -o predictions  -s scores.csv

## What else do I need?

This repository does not include code for evaluating your entry. Please see the [evaluation code repository](https://github.com/physionetchallenges/evaluation-2025) for code and instructions for evaluating your entry using the Challenge scoring metric.

## How do I learn more? How do I share more?

Please see the [Challenge website](https://physionetchallenges.org/2025/) for more details. Please post questions and concerns on the [Challenge discussion forum](https://groups.google.com/forum/#!forum/physionet-challenges). Please do not make pull requests, which may share information about your approach.

## Useful links

* [Challenge website](https://physionetchallenges.org/2025/)
* [MATLAB example code](https://github.com/physionetchallenges/matlab-example-2025)
* [Evaluation code](https://github.com/physionetchallenges/evaluation-2025)
* [Frequently asked questions (FAQ) for this year's Challenge](https://physionetchallenges.org/2025/faq/)
* [Frequently asked questions (FAQ) about the Challenges in general](https://physionetchallenges.org/faq/)
