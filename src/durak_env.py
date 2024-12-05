import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from gymnasium.spaces import Sequence, Discrete


class DurakEnv(gym.Env):
    SUITS = ['♥', '♦', '♣', '♠']
    RANKS = [6, 7, 8, 9, 10, 11, 12, 13, 14]

    def __init__(self):
        super(DurakEnv, self).__init__()

        deck = self._create_and_shuffle_deck()
        self.player_hands = [deck[:6], deck[6:12]]
        self.deck = deck[12:]
        
        self.trump_card = self.deck[-1]
        self.trump_suit = self.deck[-1][1]
        self.turn_to_attack = self._determine_first_attacker()
        self.turn_to_action = self.turn_to_attack

        self.table = []
        self.discard_pile = []
        self.game_done = False
        self.winner = None
        self.next_pass = False

        self.observation_space = spaces.Box(low=-3, high=36, shape=(50,), dtype=np.int32)
        # 5 possible actions
        self.action_space = spaces.Discrete(5)
        self.np_random = None
        # self.reset()

    def _create_and_shuffle_deck(self):
        deck = [(rank, suit) for suit in self.SUITS for rank in self.RANKS]
        random.shuffle(deck)
        return deck

    def _card_to_idx(self, card):
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
        player_hand = [self._card_to_idx(card) for card in self.player_hands[player_number]]
        opponent_hand_size = len(self.player_hands[1 - player_number])
        table = [self._card_to_idx(card[1]) for card in self.table]
        discard_pile = [self._card_to_idx(card[1]) for card in self.discard_pile]

        obs = (
            player_hand + [-1] + table + [-2] + discard_pile + [-3]
            + [opponent_hand_size, self.SUITS.index(self.trump_suit), len(self.deck), self.turn_to_attack, self.turn_to_action]
        )
        
        obs += [0] * (50 - len(obs))
        
        return np.array(obs, dtype=np.int32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random, _ = gym.utils.seeding.np_random(seed)

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
        self.next_pass = False
        
        # print("Player 0 hand:", self.player_hands[0])
        # print("Player 1 hand:", self.player_hands[1])
        # print("Trump card:", self.trump_card)
        # print("Turn to attack:", self.turn_to_attack)
        # print("Deck:", self.deck)
        # print("Example of table fulfilling: [('attack', (6, '♦')), ('defend', (12, '♦'))]")
        # print("The discard pile is a list of tables at each round.\n\n")

        return self._get_obs(self.turn_to_attack), {}

    def step(self, action):
        reward = 0
        player_number = self.turn_to_action

        if self.winner is not None:
            self.game_done = True
            reward = 200 if len(self.player_hands[player_number]) == 0 else -200
            # print(f"Looser is {self.turn_to_action}")
            return self._get_obs(player_number), reward, self.game_done, {}
        
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
            self.winner = player_number if len(self.player_hands[player_number]) == 0 else 1-player_number
            reward = 200 if len(self.player_hands[player_number]) == 0 else -200
            self.turn_to_action = 1 - self.winner
            self.turn_to_attack = self.turn_to_action

        return self._get_obs(player_number), reward, self.game_done, {'turn_to_action': self.turn_to_action}

    def _attack(self):
        """Attack - play random card from hand."""
        player_number = self.turn_to_attack
        player_cards = self.player_hands[player_number]
        
        if not player_cards:
            # print(f"Player {player_number} has no cards left to attack.")
            return -10000

        chosen_card = random.choice(player_cards)
        self.player_hands[player_number].remove(chosen_card)
        self.table.append((0, chosen_card))
        # print(f"Player {player_number}, attack {chosen_card}, +0")

        self.turn_to_action = 1 - self.turn_to_action
        return 0
    
    def get_available_actions(self):
        """Возвращает список допустимых действий в текущем состоянии."""
        if self.turn_to_action == self.turn_to_attack:
            if len(self.table)==0 and not self.next_pass:
                return [0]
            elif len(self.table)==0 and self.next_pass:
                self.next_pass = False
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
            # print(f"Player {player_number} has no cards left to attack.")
            return -10000
        
        ranks_on_table = [card[1][0] for card in self.table]
        throwable_cards = []
        for card in player_cards:
            if card[0] in ranks_on_table:
                throwable_cards.append(card)

        if not throwable_cards:
            # print(f"Player {player_number} has no cards left to throw up.")
            return -10000
        
        chosen_card = random.choice(throwable_cards)
        self.player_hands[player_number].remove(chosen_card)
        self.table.append((1, chosen_card))
        # print(f"Player {player_number}, throw up {chosen_card}, +0.5")

        self.turn_to_action = 1 - self.turn_to_action
        return 0.5

    def _defend(self):
        """Игрок защищается."""
        player_number = 1 - self.turn_to_attack
        attack_card = self.table[-1][1]
        if not self.table[-1][0] in [0, 1]:
            # print(f'Order was ruined. Last action is {self.table[-1][0]}.')
            return -10000
        
        defender_cards = self.player_hands[player_number]
        defending_cards = []
        for card in defender_cards:
            if self._is_valid_defense(card, attack_card):
                defending_cards.append(card)
        
        if not defending_cards:
            # print(f'Player {player_number} couldn\'t defend.')
            return -10000
        
        chosen_card = random.choice(defending_cards)
        self.player_hands[player_number].remove(chosen_card)
        self.table.append((3, chosen_card))
        # print(f"Player {player_number}, defend {chosen_card}, +0.5")

        self.turn_to_action = 1 - self.turn_to_action
        return 0.5

    def _pick_up(self):
        """Игрок забирает карты со стола."""
        player_number = 1 - self.turn_to_attack

        pick_up_cards = [card[1] for card in self.table]
        self.player_hands[player_number].extend(pick_up_cards)
        self.table = []
        # print(f"Player {player_number}, picks up cards from table, -2")

        self.turn_to_action = 1 - self.turn_to_action
        self.next_pass = True
        return -2

    def _pass(self):
        """Пас - раунд завершен.
        1. Пас после успешной защиты игрока
        2. Пас после того как игрок взял карты
        """
        player_number = self.turn_to_attack
        attacker_cards = self.player_hands[player_number]
        defender_cards = self.player_hands[player_number - 1]
        
        if len(self.table) == 0:
            # print(f"Player {player_number}, says PASS, +2")
            self.table = []
            self.turn_to_action = self.turn_to_attack

            # take up to 6 cards (attacker)
            attacker_take_card_number = max(6 - len(attacker_cards), 0)
            if attacker_take_card_number > 0:
                take_cards = self.deck[:attacker_take_card_number]
                self.player_hands[player_number].extend(take_cards)
                del self.deck[:attacker_take_card_number]
                # print(f"Player {player_number}, take {take_cards} from the deck, {attacker_take_card_number} needed. deck size is {len(self.deck)}")

            return 2

        elif self.table[-1][0] == 3:
            self.discard_pile.extend(self.table)
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
                # print(f"Player {player_number}, take {take_cards} from the deck, {attacker_take_card_number} needed. deck size is {len(self.deck)}")

            
            if defender_take_card_number > 0:
                take_cards = self.deck[:defender_take_card_number:]
                self.player_hands[player_number - 1].extend(take_cards)
                del self.deck[:defender_take_card_number:]
                # print(f"Player {1-player_number}, take {take_cards} from the deck, {defender_take_card_number} needed. deck size is {len(self.deck)}")

            return 0
        
    def get_turn_to_action(self, obs):
        try:
            delimiter_index = np.where(obs == -3)[0][0]
            return obs[delimiter_index + 4]
        except IndexError:
            raise ValueError("Invalid observation format. Delimiter -3 not found or incorrect observation length.")



def test_durak_env():
    env = DurakEnv()

    obs, _ = env.reset()
    print(obs)
    done = False
    total_reward_p1 = 0
    total_reward_p0 = 0
    i = 0

    while not done:
        obs = env._get_obs(player_number=env.get_turn_to_action(obs))
        player_num = env.get_turn_to_action(obs)
        available_actions = env.get_available_actions()
        
        action = np.random.choice(available_actions)
        obs, reward, done, _ = env.step(action)
        if player_num == 0: total_reward_p0 += reward
        else: total_reward_p1 += reward
        i += 1
    
    print(i, "steps")

# if __name__ == "__main__":
#     test_durak_env()

