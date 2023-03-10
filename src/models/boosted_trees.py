import os
import tensorflow_decision_forests as tfdf
import src.metadata.boosted_tree_features as boosted_tree_features


class BoostedTrees:

    name = "BoostedTreesModel"
    task = tfdf.keras.Task.CLASSIFICATION
    num_trees = 100
    features = boosted_tree_features.features
    l2_regularization = 0.4
    num_threads = os.cpu_count()
    metrics = ['accuracy', 'Precision', 'Recall']
    epochs = 1

    def save_model_diagram(self, model):
        pass

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
