from torch.utils.data import DataLoader
from config import *
from dataset.dataset import *
from statistic.collect_stat import CollectStatistics
from util.util import split_data, NodeSampler
import numpy as np
import random
from model.model import Model
from util.util import DatasetSplit

import compressed_update
import partial_participation

if device.type != 'cpu':
    torch.cuda.set_device(device)


if __name__ == "__main__":
    stat = CollectStatistics(results_eval_file_prefix=results_file_prefix)

    for seed in simulations:

        random.seed(seed)
        np.random.seed(seed)  # numpy
        torch.manual_seed(seed)  # cpu
        torch.cuda.manual_seed(seed)  # gpu
        torch.backends.cudnn.deterministic = True  # cudnn

        data_train, data_test = load_data(dataset, dataset_file_path, 'cpu')
        data_train_loader = DataLoader(data_train, batch_size=batch_size_eval, shuffle=True, num_workers=0)
        data_test_loader = DataLoader(data_test, batch_size=batch_size_eval, num_workers=0)
        dict_users = split_data(dataset, data_train, n_nodes)
        if n_nodes is None:
            n_nodes = len(dict_users)

        node_sampler = NodeSampler(n_nodes, permutation=use_permute)

        model = Model(seed, step_size, model_name=model_name, device=device, flatten_weight=True,
                      pretrained_model_file=load_model_file)

        train_loader_list = []
        dataiter_list = []
        for n in range(n_nodes):
            train_loader_list.append(
                DataLoader(DatasetSplit(data_train, dict_users[n]), batch_size=batch_size_train, shuffle=True))
            dataiter_list.append(iter(train_loader_list[n]))


        def sample_minibatch(n):
            try:
                images, labels = dataiter_list[n].next()
                if len(images) < batch_size_train:
                    dataiter_list[n] = iter(train_loader_list[n])
                    images, labels = dataiter_list[n].next()
            except StopIteration:
                dataiter_list[n] = iter(train_loader_list[n])
                images, labels = dataiter_list[n].next()

            return images, labels

        def sample_full_batch(n):
            images = []
            labels = []
            for i in range(len(train_loader_list[n].dataset)):
                images.append(train_loader_list[n].dataset[i][0])

                l = train_loader_list[n].dataset[i][1]
                if not isinstance(l, torch.Tensor):
                    l = torch.as_tensor(l)
                labels.append(l)

            return torch.stack(images), torch.stack(labels)

        w_global = model.get_weight()   # Get initial weight

        num_iter = 0
        last_output = 0
        last_amplify = 0
        last_save_latest = 0
        last_save_checkpoint = 0

        compression_method_str = compression_adaptive_method.split('-')

        # Client compression configs
        w_residual_updates_at_node = []
        compressor_at_node = []
        sum_comm_cost_at_node = []
        count_comm_at_node = []

        for n in range(n_nodes):
            sum_comm_cost_at_node.append(0.0)
            count_comm_at_node.append(0)
            w_residual_updates_at_node.append(torch.zeros(w_global.shape[0]).to(device))  # TODO: Check whether to use to(device)

            if compression_method_str[0] == 'lyapunov':
                compressor_at_node.append(compressed_update.CompressedLyapunov(node=n, target_average_cost=target_avg_comm_cost, v=lyapunov_v, init_queue=lyapunov_init_queue))
            elif compression_method_str[0] == 'fixed':
                amount_of_transmission = float(compression_method_str[1])
                compressor_at_node.append(compressed_update.CompressedNoneOrFixedRandom(node=n, target_average_cost=target_avg_comm_cost,
                                                                                        amount_of_transmission=amount_of_transmission))

        # Server compression configs
        sum_comm_cost_at_server = 0.0
        count_comm_at_server = 0
        w_residual_updates_at_server = torch.zeros(w_global.shape[0]).to(device)
        compressor_at_server = None
        if compression_method_str[0] == 'lyapunov':
            compressor_at_server = compressed_update.CompressedLyapunov(node=-1, target_average_cost=target_avg_comm_cost, v=lyapunov_v, init_queue=lyapunov_init_queue)
        elif compression_method_str[0] == 'fixed':
            amount_of_transmission = float(compression_method_str[1])
            compressor_at_server = compressed_update.CompressedNoneOrFixedRandom(node=-1, target_average_cost=target_avg_comm_cost,
                                                                                 amount_of_transmission=amount_of_transmission)

        sum_part_cost_at_node = []
        sum_obj_cost_at_node = []
        count_part_at_node = []
        part_handler_at_node = []
        for n in range(n_nodes):
            sum_part_cost_at_node.append(0.0)
            sum_obj_cost_at_node.append(0.0)
            count_part_at_node.append(0)
            if compression_method_str[0] == 'lyapunov':
                part_handler_at_node.append(partial_participation.ParticipationLyapunov(n, target_avg_participation_cost, v=lyapunov_v, init_queue=lyapunov_init_queue))
            elif compression_method_str[0] == 'fixed':
                part_handler_at_node.append(partial_participation.ParticipationStatic(n, target_avg_participation_cost))

        while True:
            print('seed', seed, '  iteration', num_iter)

            accumulated = 0

            for n in range(n_nodes):
                participation_prob = part_handler_at_node[n].get_participation(num_iter)
                if np.random.binomial(1, participation_prob) == 1:
                    model.assign_weight(w_global)
                    model.model.train()

                    for i in range(0, iters_per_round):
                        images, labels = sample_minibatch(n)

                        images, labels = images.to(device), labels.to(device)

                        if transform_train is not None:
                            images = transform_train(images)

                        model.optimizer.zero_grad()
                        output = model.model(images)
                        loss = model.loss_fn(output, labels)
                        loss.backward()
                        model.optimizer.step()

                    w_tmp = model.get_weight()  # deepcopy is already included here
                    w_tmp -= w_global  # This is the difference (i.e., update) in this round
                    sum_part_cost_at_node[n] += partial_participation.participation_cost_at_node(n, num_iter)
                else:
                    w_tmp = None

                sum_obj_cost_at_node[n] += 1.0 / participation_prob
                count_part_at_node[n] += 1
                stat.collect_stat_part_cost(seed, num_iter, n, count_part_at_node[n],
                                            sum_obj_cost_at_node[n] / count_part_at_node[n],
                                            sum_part_cost_at_node[n] / count_part_at_node[n])


                w_tmp, w_residual_updates_at_node[n] = compressor_at_node[n].get_transmitted_and_residual(num_iter, w_tmp, w_residual_updates_at_node[n])

                cost_instantaneous = compressed_update.transmission_cost_at_node(n, num_iter, w_tmp.shape[0], torch.count_nonzero(w_tmp))
                sum_comm_cost_at_node[n] += cost_instantaneous
                count_comm_at_node[n] += 1
                stat.collect_stat_comm_cost(seed, num_iter, n, count_comm_at_node[n],
                                            torch.count_nonzero(w_tmp).item(), cost_instantaneous,
                                            sum_comm_cost_at_node[n] / count_comm_at_node[n])

                if accumulated == 0:  # accumulated weights
                    w_accumulate = w_tmp
                    # Note: w_tmp cannot be used after this
                else:
                    w_accumulate += w_tmp

                accumulated += 1

            if accumulated > 0:
                w_tmp = torch.div(w_accumulate, torch.tensor(accumulated).to(device)).view(-1)
            else:
                w_tmp = torch.zeros(w_global.shape[0]).to(device)

            w_tmp, w_residual_updates_at_server = compressor_at_server.get_transmitted_and_residual(num_iter, w_tmp, w_residual_updates_at_server)

            cost_instantaneous = compressed_update.transmission_cost_at_node(-1, num_iter, w_tmp.shape[0], torch.count_nonzero(w_tmp))
            sum_comm_cost_at_server += cost_instantaneous
            count_comm_at_server += 1
            stat.collect_stat_comm_cost(seed, num_iter, -1, count_comm_at_server,
                                        torch.count_nonzero(w_tmp).item(), cost_instantaneous,
                                        sum_comm_cost_at_server / count_comm_at_server)

            w_global += w_tmp

            num_iter = num_iter + iters_per_round

            if save_checkpoint and num_iter - last_save_checkpoint >= iters_checkpoint:
                torch.save(model.model.state_dict(), save_model_file + '-checkpoint-sim-' + str(seed) + '-iter-' + str(num_iter))
                last_save_checkpoint = num_iter

            if num_iter - last_output >= min_iters_per_eval:
                stat.collect_stat_eval(seed, num_iter, model, data_train_loader, data_test_loader, w_global)
                last_output = num_iter

            if num_iter >= max_iter:
                break

        del model
        del w_global
        del w_accumulate

        torch.cuda.empty_cache()
