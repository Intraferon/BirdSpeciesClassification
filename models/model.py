from tensorflow import keras
import numpy as np
from dataset_generators.constants import *
from graphing import *
from models.context_ann import ContextANN
from PIL import Image
import tensorflow as tf
from utility import *
import shutil
import pprint


class Model:

    def __init__(self, experiment_name, model_class):

        self.experiment_name = experiment_name
        self.model_class = model_class

        self.start_stage = self.model_class.start_stage
        self.end_stage = self.model_class.end_stage

        self.model_path_list = [f"{RESOURCE_PATH}model_saves/{experiment_name}_{i}" for i in range(3)]
        self.learning_rate_schedule_state_path_list = [f"{RESOURCE_PATH}learning_rate_schedule_states/{experiment_name}_{i}.json" for i in range(3)]
        self.early_stopping_state_path_list = [f"{RESOURCE_PATH}early_stopping_states/{experiment_name}_{i}.json" for i in range(3)]
        self.early_stopping_best_weights_path_list = [f"{RESOURCE_PATH}early_stopping_best_weights/{experiment_name}_{i}.hdf5" for i in range(3)]
        self.model_log_path_list = [f"{RESOURCE_PATH}model_logs/{experiment_name}_{i}.log" for i in range(3)]
        self.model_progress_path = f"{RESOURCE_PATH}model_progress/{experiment_name}.json"
        self.model_result_path = f"{RESOURCE_PATH}model_results/{experiment_name}.json"

    def overwrite(self):
        for stage in range(self.overwrite_stage, self.end_stage + 1):
            self.overwrite_one(stage)

    def overwrite_one(self, stage):
        if os.path.exists(self.model_path_list[stage]):
            try:
                os.remove(self.model_path_list[stage])
            except PermissionError:
                shutil.rmtree(self.model_path_list[stage])
        if os.path.exists(self.learning_rate_schedule_state_path_list[stage]):
            os.remove(self.learning_rate_schedule_state_path_list[stage])
        if os.path.exists(self.early_stopping_state_path_list[stage]):
            os.remove(self.early_stopping_state_path_list[stage])
        if os.path.exists(self.early_stopping_best_weights_path_list[stage]):
            os.remove(self.early_stopping_best_weights_path_list[stage])
        if os.path.exists(self.model_log_path_list[stage]):
            os.remove(self.model_log_path_list[stage])
        if os.path.exists(self.model_result_path):
            os.remove(self.model_result_path)
        if os.path.exists(self.model_progress_path):
            progress = read_data_from_file_(self.model_progress_path)
            stage = str(stage)
            if stage in progress: progress.pop(stage)
            if progress == {}:
                os.remove(self.model_progress_path)
            else:
                save_data_to_file_(self.model_progress_path, progress)

    def get_progress(self):
        stage = None
        epoch_count = None
        converged = None
        for stage in range(self.start_stage, self.end_stage + 1):
            epoch_count, converged = self.get_one_progress(stage)
            if not converged and epoch_count < self.maximum_epoch_count[stage]:
                break
        if stage < self.end_stage and (converged or epoch_count == self.maximum_epoch_count[stage]):
            stage = stage + 1
            epoch_count = 0
            converged = False
        return stage, epoch_count, converged

    def get_one_progress(self, stage):
        epoch_count = 0
        converged = False
        if os.path.exists(self.model_progress_path):
            progress = read_data_from_file_(self.model_progress_path)
            stage = str(stage)
            if stage in progress:
                epoch_count = progress[stage]["epoch_count"]
                converged = progress[stage]["converged"]
        return epoch_count, converged

    def get_current_epoch_data(self, stage, epoch_step):
        epoch_count, converged = self.get_one_progress(stage)
        if converged:
            current_epoch_step = 0
        else:
            current_epoch_step = min(epoch_step, self.maximum_epoch_count[stage] - epoch_count)
        current_epoch_remainder = epoch_step - current_epoch_step
        return current_epoch_step, current_epoch_remainder

    def update(self, stage, model, epoch_count, converged):

        model_path_ = f"{self.model_path_list[stage]}.hdf5"
        try:
            model.save(model_path_)
        except ValueError:
            if os.path.exists(model_path_):
                os.remove(model_path_)
            model.save(self.model_path_list[stage])

        progress = {}
        if os.path.exists(self.model_progress_path):
            progress = read_data_from_file_(self.model_progress_path)
        stage = str(stage)
        if stage not in progress:
            progress[stage] = {}
        progress[stage]["epoch_count"] = epoch_count
        progress[stage]["converged"] = converged

        save_data_to_file_(self.model_progress_path, progress)

    def setup(self,
              maximum_epoch_count=None,
              overwrite_stage=None,
              train=False,
              **kwargs):

        self.instance_accuracy_metric = keras.metrics.CategoricalAccuracy()

        self.instance_top_k_accuracy_metric = keras.metrics.TopKCategoricalAccuracy(k=5)

        self.instance_mean_reciprocal_rank_metric = keras.metrics.MeanReciprocalRank()

        self.instance_loss_metric = keras.metrics.CategoricalCrossentropy()

        self.instance_confusion_matrix = {}

        if train:

            self.maximum_epoch_count = maximum_epoch_count

            self.overwrite_stage = overwrite_stage

            if self.overwrite_stage is not None:
                self.overwrite()

            self.loss_function = keras.losses.CategoricalCrossentropy()

            learning_rate_schedule_state = {stage: {} if not os.path.exists(self.learning_rate_schedule_state_path_list[stage]) else read_data_from_file_(self.learning_rate_schedule_state_path_list[stage])
                                            for stage in range(self.start_stage, self.end_stage + 1)}

            self.learning_rate_schedule = {stage: keras.callbacks.StatefulReduceLROnPlateau(self.learning_rate_schedule_state_path_list[stage],
                                                                                            monitor=kwargs["learning_rate_schedule"][stage]["monitor"],
                                                                                            factor=kwargs["learning_rate_schedule"][stage]["decay_rate"],
                                                                                            patience=kwargs["learning_rate_schedule"][stage]["patience"],
                                                                                            min_delta=kwargs["learning_rate_schedule"][stage]["min_delta"],
                                                                                            cooldown=1,
                                                                                            min_lr=kwargs["learning_rate_schedule"][stage]["end_learning_rate"])
                                           for stage in range(self.start_stage, self.end_stage + 1)}

            learning_rate = {stage: learning_rate_schedule_state[stage].get("learning_rate", kwargs["learning_rate_schedule"][stage]["initial_learning_rate"])
                             for stage in range(self.start_stage, self.end_stage + 1)}

            self.optimiser = {stage: keras.optimizers.RMSprop(learning_rate=learning_rate[stage],
                                                              rho=0.9,
                                                              momentum=0.9,
                                                              epsilon=1e-6,
                                                              clipvalue=2.0)
                              for stage in range(self.start_stage, self.end_stage + 1)}

            self.early_stopping = {stage: keras.callbacks.StatefulEarlyStopping(self.early_stopping_state_path_list[stage],
                                                                                self.early_stopping_best_weights_path_list[stage],
                                                                                monitor=kwargs["early_stopping"][stage]["monitor"],
                                                                                min_delta=kwargs["early_stopping"][stage]["min_delta"],
                                                                                patience=kwargs["early_stopping"][stage]["patience"],
                                                                                restore_best_weights=True)
                                   for stage in range(self.start_stage, self.end_stage + 1)}

            self.data_logger = {stage: keras.callbacks.CSVLogger(self.model_log_path_list[stage], append=True)
                                for stage in range(self.start_stage, self.end_stage + 1)}

        else:

            self.observation_accuracy_metric = keras.metrics.CategoricalAccuracy()

            self.observation_top_k_accuracy_metric = keras.metrics.TopKCategoricalAccuracy(k=5)

            self.observation_mean_reciprocal_rank_metric = keras.metrics.MeanReciprocalRank()

            self.observation_loss_metric = keras.metrics.CategoricalCrossentropy()

            self.observation_confusion_matrix = {}


    def load(self, stage, epoch_count):

        if epoch_count == 0:

            if stage == self.start_stage:
                model = self.model_class.create()
            else:
                model = self.get(stage - 1)

            self.model_class.alter_state(stage, model)

            model.compile(optimizer=self.optimiser[stage],
                          loss=self.loss_function,
                          metrics=[self.instance_accuracy_metric, self.instance_top_k_accuracy_metric, self.instance_mean_reciprocal_rank_metric])

        else:

            model = self.get(stage)

        return model

    def train(self, epoch_step, train_instance_generator, validation_instance_generator):

        stage, epoch_count, converged = self.get_progress()

        while not converged and epoch_step > 0:

            current_epoch_step = min(epoch_step, self.maximum_epoch_count[stage] - epoch_count)

            model = self.load(stage, epoch_count)

            model.summary(show_trainable=True)

            model.fit(x=train_instance_generator,
                      validation_data=validation_instance_generator,
                      epochs=current_epoch_step,
                      callbacks=[self.data_logger[stage],
                                 self.learning_rate_schedule[stage],
                                 self.early_stopping[stage]],
                      max_queue_size=10)

            stopped_epoch = self.early_stopping[stage].stopped_epoch
            converged = True if stopped_epoch > 0 else False

            if converged:
                epoch_step = epoch_step - (stopped_epoch - epoch_count)
                epoch_count = stopped_epoch + 1

            else:
                epoch_step = epoch_step - current_epoch_step
                epoch_count = epoch_count + current_epoch_step

            self.update(stage, model, epoch_count, converged)

            stage, epoch_count, converged = self.get_progress()

            keras.backend.clear_session()


    def test(self, test_name, stage, test_instance_generator):

        species_list = test_instance_generator.species_list

        self.instance_accuracy_metric.reset_state()
        self.instance_top_k_accuracy_metric.reset_state()
        self.instance_mean_reciprocal_rank_metric.reset_state()
        self.instance_loss_metric.reset_state()
        self.instance_confusion_matrix = {x: {z: 0 for z in species_list} for x in species_list}
        self.observation_accuracy_metric.reset_state()
        self.observation_top_k_accuracy_metric.reset_state()
        self.observation_mean_reciprocal_rank_metric.reset_state()
        self.observation_loss_metric.reset_state()
        self.observation_confusion_matrix = {x: {z: 0 for z in species_list} for x in species_list}

        model = self.get(stage)

        model.summary()

        observation_batch_size = 5000
        observation_y_prediction = np.zeros((observation_batch_size, test_instance_generator.species_count), dtype=np.float32)
        observation_y_label = None

        current_y_id = -1
        k = 0

        batch_count = test_instance_generator.__len__()

        for i in range(batch_count):

            x_input, y_id, y_label = test_instance_generator.__getitem__(i)

            y_prediction = model(x_input, training=False)

            self.instance_accuracy_metric.update_state(y_label, y_prediction)
            self.instance_top_k_accuracy_metric.update_state(y_label, y_prediction)
            self.instance_mean_reciprocal_rank_metric.update_state(y_label, y_prediction)
            self.instance_loss_metric.update_state(y_label, y_prediction)

            for j in range(y_label.shape[0]):
                instance_species_label = species_list[np.argmax(y_label[j, :])]
                instance_species_prediction = species_list[np.argmax(y_prediction[j, :])]
                self.instance_confusion_matrix[instance_species_label][instance_species_prediction] += 1

            if current_y_id == -1:
                current_y_id = y_id[0]

            for j in range(y_label.shape[0]):

                if y_id[j] != current_y_id:

                    average_y_prediction = np.mean(observation_y_prediction[:k, :], axis=0)[np.newaxis]

                    observation_species_label = species_list[np.argmax(observation_y_label)]
                    observation_species_prediction = species_list[np.argmax(average_y_prediction)]

                    self.observation_accuracy_metric.update_state(observation_y_label, average_y_prediction)
                    self.observation_top_k_accuracy_metric.update_state(observation_y_label, average_y_prediction)
                    self.observation_mean_reciprocal_rank_metric.update_state(observation_y_label, average_y_prediction)
                    self.observation_loss_metric.update_state(observation_y_label, average_y_prediction)
                    self.observation_confusion_matrix[observation_species_label][observation_species_prediction] += 1

                    k = 0

                observation_y_prediction[k] = y_prediction[j]
                observation_y_label = y_label[j][np.newaxis]
                current_y_id = y_id[j]

                k += 1

            print(f"Batch: {i} / {batch_count}")

        if k > 0:

            average_y_prediction = np.mean(observation_y_prediction[:k, :], axis=0)[np.newaxis]

            observation_species_label = species_list[np.argmax(observation_y_label)]
            observation_species_prediction = species_list[np.argmax(average_y_prediction)]

            self.observation_accuracy_metric.update_state(observation_y_label, average_y_prediction)
            self.observation_top_k_accuracy_metric.update_state(observation_y_label, average_y_prediction)
            self.observation_mean_reciprocal_rank_metric.update_state(observation_y_label, average_y_prediction)
            self.observation_loss_metric.update_state(observation_y_label, average_y_prediction)
            self.observation_confusion_matrix[observation_species_label][observation_species_prediction] += 1

            k = 0

        instance_accuracy = self.instance_accuracy_metric.result().numpy()
        instance_top_k_accuracy = self.instance_top_k_accuracy_metric.result().numpy()
        instance_mean_reciprocal_rank = self.instance_mean_reciprocal_rank_metric.result().numpy()
        instance_loss = self.instance_loss_metric.result().numpy()
        observation_accuracy = self.observation_accuracy_metric.result().numpy()
        observation_top_k_accuracy = self.observation_top_k_accuracy_metric.result().numpy()
        observation_mean_reciprocal_rank = self.observation_mean_reciprocal_rank_metric.result().numpy()
        observation_loss = self.observation_loss_metric.result().numpy()

        print(f"instance accuracy: {instance_accuracy}")
        print(f"instance top k accuracy: {instance_top_k_accuracy}")
        print(f"instance mean reciprocal rank: {instance_mean_reciprocal_rank}")
        print(f"instance loss: {instance_loss}")
        print(f"observation accuracy: {observation_accuracy}")
        print(f"observation top k accuracy: {observation_top_k_accuracy}")
        print(f"observation mean reciprocal rank: {observation_mean_reciprocal_rank}")
        print(f"observation loss: {observation_loss}")

        if os.path.exists(self.model_result_path):
            model_result_dictionary = read_data_from_file_(self.model_result_path)
        else:
            model_result_dictionary = {}

        model_result_dictionary[test_name] = {}
        model_result_dictionary[test_name]["instance_count"] = test_instance_generator.instance_count
        model_result_dictionary[test_name]["instance_accuracy"] = float(instance_accuracy)
        model_result_dictionary[test_name]["instance_top_k_accuracy"] = float(instance_top_k_accuracy)
        model_result_dictionary[test_name]["instance_mean_reciprocal_rank"] = float(instance_mean_reciprocal_rank)
        model_result_dictionary[test_name]["instance_loss"] = float(instance_loss)
        model_result_dictionary[test_name]["instance_confusion_matrix"] = self.instance_confusion_matrix
        model_result_dictionary[test_name]["observation_count"] = test_instance_generator.observation_count
        model_result_dictionary[test_name]["observation_accuracy"] = float(observation_accuracy)
        model_result_dictionary[test_name]["observation_top_k_accuracy"] = float(observation_top_k_accuracy)
        model_result_dictionary[test_name]["observation_mean_reciprocal_rank"] = float(observation_mean_reciprocal_rank)
        model_result_dictionary[test_name]["observation_loss"] = float(observation_loss)
        model_result_dictionary[test_name]["observation_confusion_matrix"] = self.observation_confusion_matrix

        save_data_to_file_(self.model_result_path, model_result_dictionary)

    def draw_graph(self, graph_type):

        log_path = self.model_log_path_list[self.end_stage]
        log_data_list = read_data_from_file_(log_path)[1:]

        x_line_list = [[] for _ in range(2)]
        y_line_list = []

        for epoch, log_data in enumerate(log_data_list):

            if graph_type == "loss":
                x_line_list[0].append(log_data[2])
                x_line_list[1].append(log_data[5])

            if graph_type == "accuracy":
                x_line_list[0].append(log_data[1])
                x_line_list[1].append(log_data[4])

            if graph_type == "top_k_accuracy":
                x_line_list[0].append(log_data[3])
                x_line_list[1].append(log_data[6])

            y_line_list.append(epoch + 1)

        parameters = {
            "x_label": "Epoch",
            "y_label": "Loss"
        }

        line_name_list = ["train", "validation"]

        grapher = Grapher(parameters)
        grapher.line_graph(line_name_list, x_line_list, [y_line_list, y_line_list])

    def get(self, stage):
        if os.path.exists(self.model_path_list[stage]):
            model = keras.models.load_model(self.model_path_list[stage])
        else:
            model = keras.models.load_model(f"{self.model_path_list[stage]}.hdf5")
        return model

    def get_path(self):
        path_ = self.model_path_list[self.end_stage]
        return path_