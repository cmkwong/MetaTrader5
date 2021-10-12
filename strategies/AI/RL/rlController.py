from production.codes import config
from production.codes.executor import mt5Model
from production.codes.data import prices
from production.codes.strategies.AI import common
import environ, models, agents, actions, experience, validation
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from production.codes.views import plotView
from production.codes.backtest import plotPre, techModel

import os
import torch
import torch.optim as optim
import numpy as np

now = datetime.now()
DT_STRING = now.strftime("%y%m%d%H%M%S")

options = {
    'main_path': "{}/projects/210215_mt5/production/docs/{}/".format(config.COMP_PATH, config.VERSION),
    'dt': DT_STRING,
    'debug': True,
}

data_options = {
    'start': (2010, 1, 2, 0, 0),
    'end': (2020, 12, 30, 0, 0),
    'symbols': ["EURUSD"],
    'timeframe': '1H',
    'timezone': "Hongkong",
    'deposit_currency': 'USD',
    'trainTestSplit': 0.7,
    'hist_bins': 100,
    'plt_save_path': os.path.join(options['main_path'], "rl_plt"),
    'debug_path': os.path.join(options['main_path'], "debug"),
    'local_min_path': os.path.join(options['main_path'], "min_data"),
    'local': False,
}

RL_options = {
    'load_net': False,
    'lr': 0.009,
    'net_file': '',
    'batch_size': 128,
    'epsilon_start': 1.0,
    'epsilon_end': 0.15,
    'gamma': 0.9,
    'reward_steps': 2,
    'net_saved_path': '',
    'val_save_path': '',
    'runs_save_path': '',
    'buffer_save_path': '',
    'replay_size': 100000,
    'monitor_buffer_size': 10000,
    'replay_init': 10000,
    'epsilon_step': 1000000,
    'target_net_sync': 1000,
    'validation_step': 30000,
    'checkpoint_step': 30000,
    'weight_visualize_step': 50000,
    'buffer_monitor_step': 100000,
    'validation_episodes': 5,
}

tech_params = {
    'ma': [5,10,25,50,100,150,200,250],
    'bb': [(20,2,2,0),(20,3,3,0),(20,4,4,0),(40,2,2,0),(40,3,3,0),(40,4,4,0)],
    'std': [(5,0),(20,0),(50,0),(100,0),(150,0),(250,0)],
    'rsi': [5,15,25,50,100,150,250],
    'stocOsci': [(5,3,3),(14,3,3),(21,14,14)],
    'macd': [(12,26,9),(19,39,9)]
}

with mt5Model.csv_Writer_Helper() as helper:
    # define loader
    prices_loader = prices.Prices_Loader(symbols=data_options['symbols'],
                                         timeframe=data_options['timeframe'],
                                         data_path=data_options['local_min_path'],
                                         start=data_options['start'],
                                         end=data_options['end'],
                                         timezone=data_options['timezone'],
                                         deposit_currency=data_options['deposit_currency'])
    # get the data
    prices_loader.get_data(data_options['local'])

    # split into train set and test set
    Train_Prices, Test_Prices = prices.split_Prices(prices_loader.Prices, percentage=data_options['trainTestSplit'])

    # # Get the technical data (training)
    # ma = techModel.get_tech_datas(Train_Prices, tech_params['ma'], tech_type='ma')
    # bb = techModel.get_tech_datas(Train_Prices, tech_params['bb'], tech_type='bb')
    # std = techModel.get_tech_datas(Train_Prices, tech_params['std'], tech_type='std')
    # rsi = techModel.get_tech_datas(Train_Prices, tech_params['rsi'], tech_type='rsi')
    # stocOsci = techModel.get_tech_datas(Train_Prices, tech_params['stocOsci'], tech_type='stocOsci')
    # macd = techModel.get_tech_datas(Train_Prices, tech_params['macd'], tech_type='macd')
    # tech_df = techModel.concat_tech_datas(join='inner', ma=ma, bb=bb, std=std, rsi=rsi, stocOsci=stocOsci, macd=macd)

    # build the env (long)
    env = environ.TechicalForexEnv(data_options['symbols'][0], Train_Prices, tech_params, True, prices_loader.all_symbols_info, 0.05, 8, 15, 1, False)
    env_val = environ.TechicalForexEnv(data_options['symbols'][0], Test_Prices, tech_params, True, prices_loader.all_symbols_info, 0.05, 8, 15, 1, False)

    net = models.SimpleFFDQN(env.get_obs_len(), env.get_action_space_size())

    # load the network
    if RL_options['load_net'] is True:
        with open(os.path.join(RL_options['net_saved_path'], RL_options['net_file']), "rb") as f:
            checkpoint = torch.load(f)
        net = models.SimpleFFDQN(env.get_obs_len(), env.get_action_space_size())
        net.load_state_dict(checkpoint['state_dict'])

    tgt_net = agents.TargetNet(net)

    # create buffer
    selector = actions.EpsilonGreedyActionSelector(RL_options['epsilon_start'])
    agent = agents.DQNAgent(net, selector)
    # agent = agents.Supervised_DQNAgent(net, selector, sample_sheet, assistance_ratio=0.2)
    exp_source = experience.ExperienceSourceFirstLast(env, agent, RL_options['gamma'], steps_count=RL_options['reward_steps'])
    buffer = experience.ExperienceReplayBuffer(exp_source, RL_options['replay_size'])

    # create optimizer
    optimizer = optim.Adam(net.parameters(), lr=RL_options['lr'])

    # create net pre-processor
    net_processor = common.netPreprocessor(net, tgt_net.target_model)

    # main training loop
    if RL_options['load_net']:
        step_idx = common.find_stepidx(RL_options['net_file'], "-", "\.")
    else:
        step_idx = 0
    eval_states = None
    best_mean_val = None

    # create the validator
    # TODO - need to modified the validator
    validator = validation.validator(env_val, net, save_path=RL_options['val_save_path'], comission=0.1)

    # create the monitor
    monitor = common.monitor(buffer, RL_options['buffer_save_path'])

    writer = SummaryWriter(log_dir=RL_options['runs_save_path'], comment="ForexRL")
    loss_tracker = common.lossTracker(writer, group_losses=100)
    with common.RewardTracker(writer, np.inf, group_rewards=100) as reward_tracker:
        while True:
            step_idx += 1
            net_processor.populate_mode(batch_size=1)
            buffer.populate(1)
            selector.epsilon = max(RL_options['epsilon_end'], RL_options['epsilon_start'] - step_idx * 0.75 / RL_options['epsilon_step'])

            new_rewards = exp_source.pop_rewards_steps()
            if new_rewards:
                reward_tracker.reward(new_rewards[0], step_idx, selector.epsilon)
            if len(buffer) < RL_options['replay_init']:
                continue

            optimizer.zero_grad()
            batch = buffer.sample(RL_options['batch_size'])

            # init the hidden both in network and tgt network
            net_processor.train_mode(batch_size=RL_options['batch_size'])
            loss_v = common.calc_loss(batch, net, tgt_net.target_model, RL_options['gamma'] ** RL_options['reward_steps'], train_on_gpu=True)
            loss_v.backward()
            optimizer.step()
            loss_value = loss_v.item()
            loss_tracker.loss(loss_value, step_idx)

            if step_idx % RL_options['target_net_sync'] == 0:
                tgt_net.sync()

            if step_idx % RL_options['checkpoint_step'] == 0:
                # idx = step_idx // CHECKPOINT_EVERY_STEP
                checkpoint = {
                    "state_dict": net.state_dict()
                }
                with open(os.path.join(RL_options['net_saved_path'], "checkpoint-{}.data".format(step_idx)), "wb") as f:
                    torch.save(checkpoint, f)

            # TODO: validation has something to changed
            if step_idx % RL_options['validation_step'] == 0:
                net_processor.eval_mode(batch_size=1)
                # writer.add_scalar("validation_episodes", validation_episodes, step_idx)
                val_epsilon = max(0, RL_options['epsilon_start'] - step_idx * 0.75 / RL_options['epsilon_step'])
                stats = validator.run(episodes=RL_options['validation_episodes'], step_idx=step_idx, epsilon=val_epsilon)
                common.valid_result_visualize(stats, writer, step_idx)

            # TODO: how to visialize the weight better? eigenvector and eigenvalues?
            if step_idx % RL_options['weight_visualize_step'] == 0:
                net_processor.eval_mode(batch_size=1)
                common.weight_visualize(net, writer)

            if step_idx % RL_options['buffer_monitor_step'] == 0:
                monitor.out_csv(RL_options['monitor_buffer_size'], step_idx)