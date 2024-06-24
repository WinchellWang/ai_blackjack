import random
import pandas as pd

class Card:
    def __init__(self, suit, value):
        self.suit = suit
        self.value = value

    def __str__(self):
        return f"{self.value} of {self.suit}"

class Deck:
    def __init__(self):
        self.cards = []
        suits = ['Hearts', 'Diamonds', 'Clubs', 'Spades']
        values = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'Jack', 'Queen', 'King', 'Ace']
        for suit in suits:
            for value in values:
                self.cards.append(Card(suit, value))

    def shuffle(self):
        random.shuffle(self.cards)

    def deal_card(self):
        if len(self.cards) == 0:
            return None
        return self.cards.pop()

class Player:
    def __init__(self, name):
        self.name = name
        self.hand = []

    def add_card_to_hand(self, card):
        self.hand.append(card)

    def hand_value(self):
        value = 0
        has_ace = False
        for card in self.hand:
            if card.value.isdigit():
                value += int(card.value)
            elif card.value in ['Jack', 'Queen', 'King']:
                value += 10
            elif card.value == 'Ace':
                has_ace = True
                value += 11
        if has_ace and value > 21:
            value -= 10
        return value

def scoreboard(players):
    # current score
    player_score = pd.DataFrame({'Score':[0.0,0.0,0.0,0.0],'Distance':[21.0,21.0,21.0,21.0],'Reward':[0.0,0.0,0.0,0.0]})
    for player in players:
        player_score.loc[player.name,'Score'] = player.hand_value()
    player_score.loc[:,'Distance'] = 10*(21 - player_score.loc[:,'Score'])/21
    player_score.loc[player_score['Distance'] < 0, 'Distance'] = -10
    return player_score

def hand(players):
    # cards on hand
    player_cards = pd.DataFrame(columns=[0,1,2,3,4,5,6,7,8], index=range(4))
    for player in players:
        m = 0
        for card in player.hand:
            player_cards.loc[player.name,m] = card.value
            m += 1
    return player_cards

def deckcard(players):
    # cards on deck (possible_from_single_player_view)
    deck_cards = pd.DataFrame(columns=['2', '3', '4', '5', '6', '7', '8', '9', '10', 'JQK', 'Ace'], index=['num', 'percent'])
    deck_cards.loc['num',:] = 4
    deck_cards.loc['num', 'JQK'] = 12
    num_cards = deck_cards.loc['num',:].sum()
    deck_cards.loc['percent',:] = deck_cards.loc['num',:]/num_cards
    for player in players:
        for card in player.hand:
            value = card.value
            if value.isdigit():
                deck_cards.loc['num',value] -= 1
            elif card.value in ['Jack', 'Queen', 'King']:
                deck_cards.loc['num', 'JQK'] -= 1
            elif card.value == 'Ace':
                deck_cards.loc['num', 'Ace'] -= 1
    num_cards = deck_cards.loc['num',:].sum()
    deck_cards.loc['percent',:] = deck_cards.loc['num',:]/num_cards
    return deck_cards

def reward(player_score):
    player_score.loc[player_score['Distance'] == -10, 'Distance'] = 100
    score_rank = player_score.sort_values('Distance')['Distance'].unique()
    rewards = [10, 7, 4, 0]
    for i, distance in enumerate(score_rank):
        player_score.loc[player_score['Distance'] == distance, 'Reward'] = rewards[i]
    player_score.loc[player_score['Distance'] == 100, 'Distance'] = -10
    player_score.loc[player_score['Distance'] == -10, 'Reward'] = 0
    return player_score

def game_init():
    # initialize game
    deck = Deck()
    deck.shuffle()
    players = [Player(0), Player(1), Player(2), Player(3)]
    # Deal two cards to each player
    for _ in range(2):
        for player in players:
            player.add_card_to_hand(deck.deal_card())
    return players, deck