import time
import os
import silence_tensorflow.auto
import tensorflow as tf
from src.models.decision_tree.random_forest.random_forest import RandomForest
from src.models.decision_tree.boosted_tree.boosted_trees import BoostedTrees
from src.models.neural_network.logistic_regression.logistic_regression import LogisticRegression
from src.models.neural_network.deep_neural_network.neural_net import NeuralNet
from src.DataManager import DataManager
from src.utils import print_run_time
from docker_info import DOCKER_PREFIX


if __name__ == '__main__':

    csv_file = DOCKER_PREFIX + 'preprocessedData.csv'
    batch_size = 160  # TRAINING + TEST + VALIDATION BATCH SIZE

    manager = DataManager(csv_file, batch_size)

    #class_obj = NeuralNet(manager)
    #class_obj = LogisticRegression(manager)
    class_obj = BoostedTrees(manager)
    #class_obj = RandomForest(manager)

    model = class_obj()

    # Load dataset using Data Manager's load_dataset function
    dataset = manager.load_dataset(csv_file, class_obj.name)

    # Split Data into Train, Test, Validation Sets
    train, validation, test = manager.get_dataset_partitions_tf(dataset)

    # Prefetch and Cache Data to help speed up training and evaluation
    train = train.cache().prefetch(tf.data.AUTOTUNE)
    validation = validation.cache().prefetch(tf.data.AUTOTUNE)
    test = test.cache().prefetch(tf.data.AUTOTUNE)
    

    # Train model using multiple epochs, with the validation
    # set being evaluated after each epoch to check for overfitting
    train_start = time.time()
    model.fit(
        train,
        epochs=class_obj.epochs,
        validation_data=validation
    )

    train_end = time.time()

    # Save Model to models directory as SavedModel instance
    model.save(class_obj.model_filepath)

    # Save model png diagram
    class_obj.save_model_diagram(model)
    #model.summary()

    # Evaluate model using test data that has been held out
    test_start = time.time()
    test_loss, test_accuracy, test_precision, test_recall = model.evaluate(test, verbose=1)
    test_end = time.time()

    print("\n Test Loss: %.3f\n Test Accuracy: %.3f\n Test Precision: %.3f\n Test Recall: %.3f\n" % (test_loss, test_accuracy, test_precision, test_recall))
    print("Training Run Time: " + print_run_time(train_end - train_start))
    print("Evaluation Run Time: " + print_run_time(test_end - test_start))
    print("Total Run Time: " + print_run_time(test_end - train_start))


