import os
import tensorflow_decision_forests as tfdf
import src.metadata.boosted_tree_features as boosted_tree_features


class RandomForest:

    name = "RandomForestModel"
    task = tfdf.keras.Task.CLASSIFICATION
    features = boosted_tree_features.features
    num_threads = os.cpu_count()
    metrics = ['accuracy', 'Precision', 'Recall']
    epochs = 1

    tuner = tfdf.tuner.RandomSearch(num_trials=20, trial_num_threads=num_threads)

    def save_model_diagram(self, model):
        pass

    def __call__(self):
        model = tfdf.keras.RandomForestModel(
            task=self.task,
            features=self.features,
            tuner=self.tuner,
            num_threads=self.num_threads,
            exclude_non_specified_features=True,
            check_dataset=False
        )

        model.compile(
            metrics=self.metrics
        )

        return model

