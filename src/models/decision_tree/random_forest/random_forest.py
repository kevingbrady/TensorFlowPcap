import os
import tensorflow_decision_forests as tfdf
from src.metadata.boosted_tree_features import features
from src.metadata.data_columns import columns
import pandas as pd
from docker_info import DOCKER_PREFIX


class RandomForest:

    name = "RandomForestModel"
    model_filepath = DOCKER_PREFIX + '/home/kgb/PycharmProjects/TensorFlowPcap/src/models/decision_tree/random_forest/RandomForestModel'
    task = tfdf.keras.Task.CLASSIFICATION
    feature_names = []
    num_threads = os.cpu_count()
    num_trees = 26
    metrics = ['accuracy', 'Precision', 'Recall']
    epochs = 1

    def __init__(self, manager):

        if not os.path.exists(self.model_filepath):
            os.mkdir(self.model_filepath)

        self.feature_names = manager.feature_names

    def save_model(self, model):
        model.save(self.model_filepath)

    def save_model_diagram(self, model):

        with open(DOCKER_PREFIX + 'src/models/decision_tree/random_forest/' + self.name + '.html', 'w+') as f:
            f.write(
                tfdf.model_plotter.plot_model(
                    model,
                    max_depth=self.num_trees
                )
            )

    def __call__(self):

        #tuner = tfdf.tuner.RandomSearch(num_trials=20, trial_num_threads=self.num_threads)

        model = tfdf.keras.RandomForestModel(
            name=self.name,
            task=tfdf.keras.Task.CLASSIFICATION,
            features=features,
            verbose=1,
            #tuner=self.tuner,
            num_threads=self.num_threads,
            num_trees=self.num_trees,
            pure_serving_model=True,
            exclude_non_specified_features=True,
            check_dataset=False
        )

        model.compile(
            metrics=self.metrics
        )

        return model
