import time
import silence_tensorflow.auto
import tensorflow as tf
import keras
from src.models.decision_tree.random_forest.random_forest import RandomForest
from src.models.decision_tree.boosted_tree.boosted_trees import BoostedTrees
from src.models.neural_network.logistic_regression.logistic_regression import LogisticRegression
from src.models.neural_network.deep_neural_network.neural_net import NeuralNet
from src.DataManager import DataManager
from src.utils import print_run_time, print_results
from docker_info import DOCKER_PREFIX


if __name__ == '__main__':

    csv_file = DOCKER_PREFIX + 'preprocessedData.csv'
    batch_size = 120  # TRAINING + TEST + VALIDATION BATCH SIZE      (40 (32), 80 (64), 120 (96), 160 (128))

    manager = DataManager(csv_file, batch_size)

    class_obj = NeuralNet(manager)
    #class_obj = LogisticRegression(manager)
    #class_obj = BoostedTrees(manager)
    #class_obj = RandomForest(manager)

    # Load dataset using Data Manager's load_dataset function
    dataset = manager.load_dataset(csv_file, class_obj.name)

    # Split Data into Train, Test, Validation Sets
    train, validation, test = manager.get_dataset_partitions_tf(dataset)

    # Prefetch and Cache Data to help speed up training and evaluation
    train = train.cache().prefetch(tf.data.AUTOTUNE)
    validation = validation.cache().prefetch(tf.data.AUTOTUNE)
    test = test.cache().prefetch(tf.data.AUTOTUNE)

    model = class_obj()
    

    # Train model using multiple epochs, with the validation
    # set being evaluated after each epoch to check for overfitting
    train_start = time.time()

    model.fit(
        train,
        epochs=class_obj.epochs,
        #steps_per_epoch=manager.training_steps,
        validation_data=validation,
        #validation_steps=manager.validation_steps,
        verbose=0
    )


    train_end = time.time()

    # Save Model to models directory as SavedModel instance
    class_obj.save_model(model)
    #model.save(class_obj.model_filepath)

    # Save model png diagram
    class_obj.save_model_diagram(model)
    #model.summary()

    # Evaluate model using test data that has been held out
    test_start = time.time()
    results = model.evaluate(test, return_dict=True, verbose=0)
    test_end = time.time()

    print("Test Dataset")
    print({print_results(x, y) for x, y in results.items()})
    print("Training Run Time: " + print_run_time(train_end - train_start))
    print("Evaluation Run Time: " + print_run_time(test_end - test_start))
    print("Total Run Time: " + print_run_time(test_end - train_start))


