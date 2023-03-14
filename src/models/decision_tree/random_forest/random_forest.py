import os
import tensorflow as tf
import tensorflow_decision_forests as tfdf
import src.metadata.boosted_tree_features as boosted_tree_features
import dtreeviz
import pandas as pd


class RandomForest:
    name = "RandomForestModel"
    task = tfdf.keras.Task.CLASSIFICATION
    features = boosted_tree_features.features
    feature_names = []
    num_threads = os.cpu_count()
    num_trees = 50
    metrics = ['accuracy', 'Precision', 'Recall']
    epochs = 1

    tuner = tfdf.tuner.RandomSearch(num_trials=20, trial_num_threads=num_threads)

    def __init__(self, manager):

        self.feature_names = manager.feature_names

    def save_model_diagram(self, model):

        '''
        dataset = pd.read_csv('/home/kgb/PycharmProjects/TensorFlowPcap/preprocessedData.csv')   #chunksize=100000)

        model_visualization = dtreeviz.model(
                                             model,
                                             tree_index=2,
                                             X_train=dataset[lambda x: x != self.feature_names[-1]],
                                             y_train=dataset[self.feature_names[-1]],
                                             feature_names=self.feature_names,
                                             target_name=self.feature_names[-1]
                            )
        viewer = model_visualization.view(orientation='LR', scale=1.2)
        viewer.save(self.name + '.svg')
        viewer.show()
        '''
        with open('src/models/decision_tree/random_forest/' + self.name + '.html', 'w+') as f:
            f.write(tfdf.model_plotter.plot_model(
                model,
                max_depth=self.num_trees
            )
            )

    def __call__(self):
        model = tfdf.keras.RandomForestModel(
            task=self.task,
            features=self.features,
            tuner=self.tuner,
            num_threads=self.num_threads,
            num_trees=self.num_trees,
            exclude_non_specified_features=True,
            check_dataset=False
        )

        model.compile(
            metrics=self.metrics
        )

        return model
