import gym
from gym import spaces
import numpy as np
import random


class DurakEnv(gym.Env):
    SUITS = ['♥', '♦', '♣', '♠']
    RANKS = [6, 7, 8, 9, 10, 11, 12, 13, 14]

    def __init__(self):
        super(DurakEnv, self).__init__()

        self.deck = self._create_and_shuffle_deck()
        self.player_hands = [self.deck[:6], self.deck[6:12]]
        self.deck = self.deck[12:]
        
        self.trump_card = self.deck[-1]
        self.trump_suit = self.deck[-1][1]
        self.turn_to_attack = self._determine_first_attacker()
        self.turn_to_action = self.turn_to_attack

        self.table = []
        self.discard_pile = []
        self.game_done = False
        self.winner = None

        # Карты как ранги от 0 до 35 (9 карт x 4 масти)
        self.observation_space = spaces.Dict({
            "player_hand": spaces.MultiDiscrete([36] * 6),
            "opponent_hand_size": spaces.Discrete(36),
            "table": spaces.MultiDiscrete([36] * 12),
            "discard_pile": spaces.MultiDiscrete([36] * 36),
            "trump_suit": spaces.Discrete(4),
            "deck_size": spaces.Discrete(24),
            "turn_to_action": spaces.Discrete(2),
            "turn_to_attack": spaces.Discrete(2)
        })

        # 5 possible actions
        self.action_space = spaces.Discrete(5)
        # self.reset()

    def _create_and_shuffle_deck(self):
        deck = [(rank, suit) for suit in self.SUITS for rank in self.RANKS]
        random.shuffle(deck)
        return deck

    def _encode_card(self, card):
        """Кодирование карты (ранг от 0 до 35)"""
        rank_index = self.RANKS.index(card[0])
        suit_index = self.SUITS.index(card[1])
        return rank_index + suit_index * len(self.RANKS)

    def _determine_first_attacker(self):
        player0_hand = self.player_hands[0]
        player1_hand = self.player_hands[1]

        player0_trumps = [card for card in player0_hand if card[1] == self.trump_suit]
        player1_trumps = [card for card in player1_hand if card[1] == self.trump_suit]

        player0_lowest_trump = min(player0_trumps, default=(float('inf'), None))
        player1_lowest_trump = min(player1_trumps, default=(float('inf'), None))

        if player0_lowest_trump[0] < player1_lowest_trump[0]:
            return 0
        else:
            return 1
    
    def _get_obs(self, player_number):
        player_hand = self.player_hands[player_number]
        opponent_hand_size = len(self.player_hands[1 - player_number])

        return {
            "player_hand": player_hand, # [self._encode_card(card) for card in player_hand],
            "opponent_hand_size": opponent_hand_size,
            "table": self.table, #[self._encode_card(card[1]) for card in self.table],
            "discard_pile": self.discard_pile, # [self._encode_card(card[1]) for card in self.discard_pile],
            "trump_suit": self.trump_suit, #self.SUITS.index(self.trump_suit),
            "deck_size": len(self.deck),
            "turn_to_attack": self.turn_to_attack,
            "turn_to_action": self.turn_to_action,
        }

    def reset(self):
        self.deck = self._create_and_shuffle_deck()
        self.player_hands = [self.deck[:6], self.deck[6:12]]
        self.deck = self.deck[12:]
        
        self.trump_card = self.deck[-1]
        self.trump_suit = self.deck[-1][1]
        self.turn_to_attack = self._determine_first_attacker()
        self.turn_to_action = self.turn_to_attack

        self.table = []
        self.discard_pile = []
        self.game_done = False
        self.winner = None
        
        print(self.player_hands[0])
        print(self.player_hands[1])
        print(self.trump_suit)
        print(self.turn_to_attack)
        return self._get_obs(self.turn_to_attack)

    def step(self, action):
        reward = 0
        player_number = self.turn_to_action

        # check that the action is valid
        available_actions = self.get_available_actions()
        if action not in available_actions:
            raise ValueError(f"Action {action} is not valid in this state.")

        if self.winner is not None:
            self.game_done = True
            reward = 200 if len(self.player_hands[player_number]) == 0 else -200
            print(f"Looser is {self.turn_to_action}")
            return self._get_obs(player_number), reward, self.game_done, {"available_actions": available_actions}
        
        if action == 0:
            if len(self.table) == 0:
                reward = self._attack()
            else:
                raise ValueError(f"Action {action} is not valid in this state. You can attack only on the first action in the round.")
        elif action == 1:
            reward = self._throw_card()
        elif action == 2:
            reward = self._pass()
        elif action == 3:
            reward = self._defend()
        elif action == 4:
            reward = self._pick_up()
        else:
            raise ValueError(f"Action {action} is not valid in this state. Action must be in the range [0,4]. ")

        # check for the end of a game
        if (len(self.player_hands[0]) == 0 or len(self.player_hands[1]) == 0) and not self.game_done:
            # self.game_done = True
            self.winner = player_number if len(self.player_hands[player_number]) == 0 else 1-player_number
            reward = 200 if len(self.player_hands[player_number]) == 0 else -200
            print(f"Winner is {self.winner}")
            self.turn_to_action = 1 - self.winner
            self.turn_to_attack = self.turn_to_action

        return self._get_obs(player_number), reward, self.game_done, {"available_actions": available_actions}

    def _attack(self):
        """Attack - play random card from hand."""
        player_number = self.turn_to_attack
        player_cards = self.player_hands[player_number]
        
        if not player_cards:
            print(f"Player {player_number} has no cards left to attack.")
            return -10000

        chosen_card = random.choice(player_cards)
        self.player_hands[player_number].remove(chosen_card)
        self.table.append((0, chosen_card))
        print(f"Player {player_number}, attack {chosen_card}, +0")

        self.turn_to_action = 1 - self.turn_to_action
        return 0
    
    def get_available_actions(self):
        """Возвращает список допустимых действий в текущем состоянии."""
        if self.turn_to_action == self.turn_to_attack:
            if not self.table:
                return [0]
            elif self.table == [(4, None)]:
                return [2]
            else:
                return [1, 2] if self._can_throw_card() else [2]
        else:
            if self._can_defend():
                return [3, 4]
            else:
                return [4]

    def _can_defend(self):
        """Проверяет, есть ли у игрока возможность защититься."""
        defender_cards = self.player_hands[1 - self.turn_to_attack]
        attack_card = self.table[-1][1]
        for card in defender_cards:
            if self._is_valid_defense(card, attack_card):
                return True
        return False
    
    def _is_valid_defense(self, card, attack_card):
        if card[1] == self.trump_suit:
            if attack_card[1] == self.trump_suit and card[0] < attack_card[0]:
                return False
            else:
                return True
        else:
            if card[1] == attack_card[1] and card[0] > attack_card[0]:
                return True
            else:
                return False

    def _can_throw_card(self):
        """Проверяет, есть ли возможность подкинуть карту."""
        ranks_on_table = [card[1][0] for card in self.table]
        for card in self.player_hands[self.turn_to_attack]:
            if card[0] in ranks_on_table:
                return True
        return False

    def _throw_card(self):
        """Подкидывает рандомную допустимую карту если такая есть."""
        player_number = self.turn_to_attack
        player_cards = self.player_hands[player_number]
        
        if not player_cards:
            print(f"Player {player_number} has no cards left to attack.")
            return -10000
        
        ranks_on_table = [card[1][0] for card in self.table]
        throwable_cards = []
        for card in player_cards:
            if card[0] in ranks_on_table:
                throwable_cards.append(card)

        if not throwable_cards:
            print(f"Player {player_number} has no cards left to throw up.")
            return -10000
        
        chosen_card = random.choice(throwable_cards)
        self.player_hands[player_number].remove(chosen_card)
        self.table.append((1, chosen_card))
        print(f"Player {player_number}, throw up {chosen_card}, +0.5")

        self.turn_to_action = 1 - self.turn_to_action
        return 0.5

    def _defend(self):
        """Игрок защищается."""
        player_number = 1 - self.turn_to_attack
        attack_card = self.table[-1][1]
        if not self.table[-1][0] in [0, 1]:
            print(f'Order was ruined. Last action is {self.table[-1][0]}.')
            return -10000
        
        defender_cards = self.player_hands[player_number]
        defending_cards = []
        for card in defender_cards:
            if self._is_valid_defense(card, attack_card):
                defending_cards.append(card)
        
        if not defending_cards:
            print(f'Player {player_number} couldn\'t defend.')
            return -10000
        
        chosen_card = random.choice(defending_cards)
        self.player_hands[player_number].remove(chosen_card)
        self.table.append((3, chosen_card))
        print(f"Player {player_number}, defend {chosen_card}, +0.5")

        self.turn_to_action = 1 - self.turn_to_action
        return 0.5

    def _pick_up(self):
        """Игрок забирает карты со стола."""
        player_number = 1 - self.turn_to_attack

        pick_up_cards = [card[1] for card in self.table]
        self.player_hands[player_number].extend(pick_up_cards)
        self.table = [(4, None)]
        print(f"Player {player_number}, picks up cards from table, -2")

        self.turn_to_action = 1 - self.turn_to_action
        return -2

    def _pass(self):
        """Пас - раунд завершен.
        1. Пас после успешной защиты игрока
        2. Пас после того как игрок взял карты
        """
        player_number = self.turn_to_attack
        attacker_cards = self.player_hands[player_number]
        defender_cards = self.player_hands[player_number - 1]
        last_action = self.table[-1]
        
        if last_action[0] == 4:
            print(f"Player {player_number}, says PASS, +2")
            self.table = []
            self.turn_to_action = self.turn_to_attack

            # take up to 6 cards (attacker)
            attacker_take_card_number = max(6 - len(attacker_cards), 0)
            if attacker_take_card_number > 0:
                take_cards = self.deck[len(self.deck)-attacker_take_card_number:]
                self.player_hands[player_number].extend(take_cards)
                del self.deck[len(self.deck)-attacker_take_card_number:]
                print(f"Player {player_number}, take {take_cards} from the deck, {attacker_take_card_number} needed. deck size is {len(self.deck)}")

            return 2

        elif last_action[0] == 3:
            print(f"Player {player_number}, says PASS, +0")
            self.discard_pile.append(self.table)
            self.table = []
            self.turn_to_attack = 1 - self.turn_to_attack
            self.turn_to_action = self.turn_to_attack

            # take up to 6 cards (both players, first attacker)
            attacker_take_card_number = max(6 - len(attacker_cards), 0)
            defender_take_card_number = max(6 - len(defender_cards), 0)

            if attacker_take_card_number + defender_take_card_number > len(self.deck):
                take_card_to_equal = min(abs(len(attacker_cards) - len(defender_cards)), len(self.deck))
                take_cards = self.deck[:take_card_to_equal]
                del self.deck[:take_card_to_equal:]

                if len(attacker_cards) < len(defender_cards):
                    self.player_hands[player_number].extend(take_cards)
                else:
                    self.player_hands[player_number - 1].extend(take_cards)

                take_cards = self.deck[:len(self.deck)//2:]
                self.player_hands[player_number].extend(take_cards)
                del self.deck[:len(self.deck)//2:]

                self.player_hands[player_number - 1].extend(self.deck)
                self.deck = []
                return 0


            if attacker_take_card_number > 0:
                take_cards = self.deck[:attacker_take_card_number:]
                self.player_hands[player_number].extend(take_cards)
                del self.deck[:attacker_take_card_number:]
                print(f"Player {player_number}, take {take_cards} from the deck, {attacker_take_card_number} needed. deck size is {len(self.deck)}")

            
            if defender_take_card_number > 0:
                take_cards = self.deck[:defender_take_card_number:]
                self.player_hands[player_number - 1].extend(take_cards)
                del self.deck[:defender_take_card_number:]
                print(f"Player {1-player_number}, take {take_cards} from the deck, {defender_take_card_number} needed. deck size is {len(self.deck)}")

            return 0



def test_durak_env():
    env = DurakEnv()
    num_episodes = 1
    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        total_reward_p1 = 0
        total_reward_p0 = 0
        print(f"\n=== Начало эпизода {episode + 1} ===")
        i = 0
        t = 0
        while not done:
            # Получение доступных действий
            obs = env._get_obs(player_number=obs['turn_to_action'])
            player_num = obs['turn_to_action']
            available_actions = env.get_available_actions()
            if i < t:
                print("---начало хода---")
                print(obs)
                print(f"Текущая рука: {obs['player_hand']}, Доступные действия: {available_actions}")
            
            # Случайное действие из доступных
            action = np.random.choice(available_actions)
            obs, reward, done, info = env.step(action)
            # print(reward)
            if player_num == 0: total_reward_p0 += reward
            else: total_reward_p1 += reward

            if i < t:
                print(f"Действие: {action}, Награда: {reward}, Завершено: {done}, Колода: {obs['deck_size']}")
                print(f"Стол: {obs['table']}, Количество карт противника: {obs['opponent_hand_size']}")
                print(obs)
                print("---конец хода---\n")
            i += 1
        
        print(i)
        print(f"=== Конец эпизода {episode + 1}, Общая награда: {total_reward_p0, total_reward_p1} ===")

if __name__ == "__main__":
    test_durak_env()

