import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import *


class CollectStatistics:
    def __init__(self, results_eval_file_prefix=os.path.dirname(__file__) + '/results'):
        self.results_eval_file_name = results_eval_file_prefix + '_eval.csv'
        self.results_cost_file_name_clients = results_eval_file_prefix + '_client_to_server_comm_cost.csv'
        self.results_cost_file_name_server = results_eval_file_prefix + '_server_to_client_comm_cost.csv'
        self.results_cost_file_name_participation = results_eval_file_prefix + '_computation_cost.csv'

        with open(self.results_eval_file_name, 'a') as f:
            f.write(
                'sim_seed,num_iter,training_loss,training_accuracy,test_accuracy\n')
            f.close()

        for filename in [self.results_cost_file_name_clients, self.results_cost_file_name_server]:
            with open(filename, 'a') as f:
                f.write(
                    'sim_seed,num_iter,node,count,transmitted_elements,cost_instantaneous,cost_avg\n')
                f.close()

        with open(self.results_cost_file_name_participation, 'a') as f:
            f.write(
                'sim_seed,num_iter,node,count,obj_avg,cost_avg\n')
            f.close()

    def collect_stat_eval(self, seed, num_iter, model, train_data_loader, test_data_loader, w_global=None):
        loss_value, train_accuracy = model.accuracy(train_data_loader, w_global, device)
        _, prediction_accuracy = model.accuracy(test_data_loader, w_global, device)

        print("Simulation seed", seed, "Iteration", num_iter,
              "Training accuracy", train_accuracy, "Testing accuracy", prediction_accuracy)

        with open(self.results_eval_file_name, 'a') as f:
            f.write(str(seed) + ',' + str(num_iter) + ',' + str(loss_value) + ','
                    + str(train_accuracy) + ',' + str(prediction_accuracy) + '\n')
            f.close()

    def collect_stat_comm_cost(self, seed, num_iter, node, count, transmitted_elements, cost_instantaneous, cost_avg):
        if node < 0:
            filename = self.results_cost_file_name_server
        else:
            filename = self.results_cost_file_name_clients
        with open(filename, 'a') as f:
            f.write(str(seed) + ',' + str(num_iter) + ',' + str(node) + ',' + str(count)
                    + ',' + str(transmitted_elements) + ',' + str(cost_instantaneous)
                    + ',' + str(cost_avg) + '\n')
            f.close()

    def collect_stat_part_cost(self, seed, num_iter, node, count, obj_avg, cost_avg):
        filename = self.results_cost_file_name_participation
        with open(filename, 'a') as f:
            f.write(str(seed) + ',' + str(num_iter) + ',' + str(node) + ',' + str(count)
                    + ',' + str(obj_avg) + ',' + str(cost_avg) + '\n')
            f.close()
