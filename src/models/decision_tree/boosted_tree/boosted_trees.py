import os
import tensorflow_decision_forests as tfdf
import src.metadata.boosted_tree_features as boosted_tree_features


class BoostedTrees:
    name = "BoostedTreesModel"
    task = tfdf.keras.Task.CLASSIFICATION
    num_trees = 20
    features = boosted_tree_features.features
    feature_names = []
    l2_regularization = 0.4
    num_threads = os.cpu_count()
    metrics = ['accuracy', 'Precision', 'Recall']
    epochs = 1

    def __init__(self, manager):
        self.feature_names = manager.feature_names

    def save_model_diagram(self, model):

        with open('src/models/decision_tree/boosted_tree/' + self.name + '.html', 'w+') as f:
            f.write(tfdf.model_plotter.plot_model(
                                model,
                                max_depth=self.num_trees
                        )
            )

    def __call__(self):
        model = tfdf.keras.GradientBoostedTreesModel(
            task=self.task,
            num_trees=self.num_trees,
            features=self.features,
            l2_regularization=self.l2_regularization,
            num_threads=self.num_threads,
            exclude_non_specified_features=True,
            check_dataset=False
        )

        model.compile(
            metrics=self.metrics
        )

        return model
