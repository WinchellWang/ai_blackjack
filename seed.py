from sklearn.model_selection import train_test_split
import pandas as pd
from blackjack import *
from neural_network import *

def init_training():
    log = pd.DataFrame(columns=['num', 'player',
                                'distance', 'reward',
                                'o0_distance', 'o0_reward',
                                'o1_distance', 'o1_reward',
                                'o2_distance', 'o2_reward',
                                'hit', 'hold', 'win'])
    row = 0
    for game_round in range(500):
        players, deck = game_init()
        for player in players:
            for _ in range(14):
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
                o_player = 0
                for other_player in players:
                    if other_player.name != player.name:
                        log.loc[row,'o%i_distance' % o_player] = player_score.loc[other_player.name,'Distance']
                        log.loc[row,'o%i_reward' % o_player] = player_score.loc[other_player.name,'Reward']/10
                        o_player += 1
                if player.hand_value() > 21:
                    log.loc[row,'hold'] = 1
                    row += 1
                    break
                # 1 hit, 0 stand
                if random.randrange(2) == 1:
                    log.loc[row,'hit'] = 1
                    player.add_card_to_hand(deck.deal_card())
                    row += 1
                else:
                    log.loc[row,'hold'] = 1
                    row += 1
                    break
        final = reward(scoreboard(players))
        winner_list = final.loc[final['Reward'] == 10].index.to_list()
        for winner in winner_list:
            log.loc[(log['num'] == game_round) & (log['player'] == winner), 'win'] = 1
    return log

log = init_training()
log.to_csv('random_training_init.csv', index=False)
initial_data = pd.read_csv('random_training_init.csv')
x = initial_data.loc[:,['distance', 'reward',
                        'o0_distance', 'o0_reward',
                        'o1_distance', 'o1_reward',
                        'o2_distance', 'o2_reward',]].copy()
y = initial_data.loc[:,['hit', 'hold', 'win']].copy()
x_1, x_2, y_1, y_2 = train_test_split(x, y, test_size=0.5, random_state=60)
x_1, x_3, y_1, y_3 = train_test_split(x_1, y_1, test_size=0.5, random_state=14)
x_2, x_4, y_2, y_4 = train_test_split(x_2, y_2, test_size=0.5, random_state=9)
model_0 = model_training(x_1,y_1,num_epochs=2000)
model_1 = model_training(x_2,y_2,num_epochs=2000)
model_2 = model_training(x_3,y_3,num_epochs=2000)
model_3 = model_training(x_4,y_4,num_epochs=2000)
model = {0:model_0, 1:model_1, 2:model_2, 3:model_3}
torch.save(model, 'model_init.pt')