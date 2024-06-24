from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from blackjack import *
from neural_network import *

def ai_game(model, loop_num = 500):
    log = pd.DataFrame(columns=['num', 'player',
                                'distance', 'reward',
                                'o0_distance', 'o0_reward',
                                'o1_distance', 'o1_reward',
                                'o2_distance', 'o2_reward',
                                'hit', 'hold', 'win'])
    row = 0
    for game_round in range(loop_num):
        players, deck = game_init()
        for player in players:
            while player.hand_value() <= 21:
                # update log
                player_score = scoreboard(players)
                player_score = reward(player_score)
                log.loc[row,'num'] = game_round
                log.loc[row,'player'] = player.name
                log.loc[row,'distance'] = player_score.loc[player.name,'Distance']
                log.loc[row,'reward'] = player_score.loc[player.name,'Reward']
                log.loc[row,'win'] = 0
                log.loc[row,'hit'] = 0
                log.loc[row,'hold'] = 0
                pred_input = np.array([[
                    log.loc[row,'distance'], # distance
                    log.loc[row,'reward'], # reward
                    np.nan, # o0_distance
                    np.nan, # o0_reward /10
                    np.nan, # o1_distance
                    np.nan, # o1_reward /10
                    np.nan, # o2_distance
                    np.nan, # o2_reward /10
                    ]])
                o_player = 0
                for other_player in players:
                    if other_player.name != player.name:
                        log.loc[row,'o%i_distance' % o_player] = player_score.loc[other_player.name,'Distance']
                        log.loc[row,'o%i_reward' % o_player] = player_score.loc[other_player.name,'Reward']/10
                        pred_input[0,o_player*2+2] = log.loc[row,'o%i_distance' % o_player]
                        pred_input[0,o_player*2+3] = log.loc[row,'o%i_reward' % o_player]
                        o_player += 1
                prediction = model_prediction(pred_input,model[player.name])
                if prediction[0,0] > prediction[0,1]:
                    player.add_card_to_hand(deck.deal_card())
                    log.loc[row,'hit'] = 1
                    row += 1
                else:
                    log.loc[row,'hold'] = 1
                    row += 1
                    break
            if player.hand_value() > 21:
                player_score = scoreboard(players)
                player_score = reward(player_score)
                log.loc[row,'num'] = game_round
                log.loc[row,'player'] = player.name
                log.loc[row,'distance'] = player_score.loc[player.name,'Distance']
                log.loc[row,'reward'] = player_score.loc[player.name,'Reward']
                log.loc[row,'win'] = 0
                log.loc[row,'hit'] = 0
                log.loc[row,'hold'] = 1
                o_player = 0
                for other_player in players:
                    if other_player.name != player.name:
                        log.loc[row,'o%i_distance' % o_player] = player_score.loc[other_player.name,'Distance']
                        log.loc[row,'o%i_reward' % o_player] = player_score.loc[other_player.name,'Reward']/10
                        o_player += 1
                row += 1
        # winner 1, loser 0
        final = reward(scoreboard(players))
        winner_list = final.loc[final['Reward'] == 10].index.to_list()
        for winner in winner_list:
            log.loc[(log['num'] == game_round) & (log['player'] == winner), 'win'] = 1
    return log

def ai_train(training_data):
    x = training_data.loc[:,['distance', 'reward',
                            'o0_distance', 'o0_reward',
                            'o1_distance', 'o1_reward',
                            'o2_distance', 'o2_reward',]].copy()
    y = training_data.loc[:,['hit', 'hold', 'win']].copy()
    x_1, x_2, y_1, y_2 = train_test_split(x, y, test_size=0.5, random_state=60)
    x_1, x_3, y_1, y_3 = train_test_split(x_1, y_1, test_size=0.5, random_state=14)
    x_2, x_4, y_2, y_4 = train_test_split(x_2, y_2, test_size=0.5, random_state=9)
    model_0 = model_training(x_1,y_1,num_epochs=2000)
    model_1 = model_training(x_2,y_2,num_epochs=2000)
    model_2 = model_training(x_3,y_3,num_epochs=2000)
    model_3 = model_training(x_4,y_4,num_epochs=2000)
    model = {0:model_0, 1:model_1, 2:model_2, 3:model_3}
    return model

def get_top_winners(log, n=2):
    log_unique = log.drop_duplicates(subset=['num', 'player'])
    top_winners = log_unique[log_unique['win'] == 1]['player'].value_counts().nlargest(n).index.tolist()
    return top_winners

def ai_update(model,training_log,epochs=2000):
    winner_list = get_top_winners(training_log, n=2)
    x = training_log.loc[:,['distance', 'reward',
                        'o0_distance', 'o0_reward',
                        'o1_distance', 'o1_reward',
                        'o2_distance', 'o2_reward',]].copy()
    y = training_log.loc[:,['hit', 'hold', 'win']].copy()
    x_1, x_2, y_1, y_2 = train_test_split(x, y, test_size=0.5, random_state=60)
    x_1, x_3, y_1, y_3 = train_test_split(x_1, y_1, test_size=0.5, random_state=14)
    x_2, x_4, y_2, y_4 = train_test_split(x_2, y_2, test_size=0.5, random_state=9)
    model_0 = continue_training(model[winner_list[0]],x_1,y_1,num_epochs=epochs)
    model_1 = continue_training(model[winner_list[0]],x_2,y_2,num_epochs=epochs)
    model_2 = continue_training(model[winner_list[1]],x_3,y_3,num_epochs=epochs)
    model_3 = continue_training(model[winner_list[1]],x_4,y_4,num_epochs=epochs)
    model = {0:model_0, 1:model_1, 2:model_2, 3:model_3}
    return model