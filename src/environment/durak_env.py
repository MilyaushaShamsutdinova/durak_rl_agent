# import gym
# import random
# import numpy as np
# from gym import spaces

import gymnasium as gym
import random
import numpy as np
from gymnasium import spaces


class DurakEnv(gym.Env):
    def __init__(self):
        super(DurakEnv, self).__init__()

        # Action space: The player can play a card from hand or pass/take cards.
        # The action is indexed by the card position in hand + additional actions (pass/take).
        self.action_space = spaces.Discrete(7)  # 6 cards in hand + 1 for "pass/take"

        # Observation space:
        # - Player hand (6 cards)
        # - Opponent's hand size (hidden)
        # - Board state (cards on the board)
        # - Trump suit
        self.observation_space = spaces.Dict({
            "player_hand": spaces.MultiDiscrete([9]*6),  # 6 cards, 9 possible values (6-Ace)
            "opponent_hand_size": spaces.Discrete(7),    # Opponent hand size (0-6)
            "board_state": spaces.MultiDiscrete([9, 9]),  # Attacker card, defender card (can extend for multiple attacks)
            "trump_suit": spaces.Discrete(4),            # Trump suit (0-3 for 4 suits)
        })

        # Game state
        self.deck = []
        self.player_hand = []
        self.opponent_hand = []
        self.board = []  # Cards currently in play (attack and defense)
        self.trump_card = None
        self.trump_suit = None
        self.player_turn = None
        self.done = False

        self._initialize_game()

    def _initialize_game(self):
        """ Initialize the game state by creating deck, dealing cards, and setting up trump card. """
        deck = self.create_and_shuffle_deck()

        # Deal cards
        self.player_hand = deck[:6]
        self.opponent_hand = deck[6:12]
        self.deck = deck[12:]

        # Select trump card
        self.trump_card = self.deck.pop(0)
        self.trump_suit = self.trump_card[1]

        # Determine first attacker based on trump card rank
        self.player_turn = self._determine_first_attacker()

        # Initialize board (empty at the start)
        self.board = []

    def create_and_shuffle_deck(self):
        """Create a deck of 36 cards (6 to Ace in four suits) and shuffle it."""
        suits = ['hearts', 'diamonds', 'clubs', 'spades']
        ranks = [6, 7, 8, 9, 10, 11, 12, 13, 14]
        deck = [(rank, suit) for suit in suits for rank in ranks]
        random.shuffle(deck)
        return deck

    def _determine_first_attacker(self):
        """Determine the first attacker based on the lowest trump card."""
        player_trumps = [card for card in self.player_hand if card[1] == self.trump_suit]
        opponent_trumps = [card for card in self.opponent_hand if card[1] == self.trump_suit]

        player_lowest_trump = min(player_trumps, default=None)
        opponent_lowest_trump = min(opponent_trumps, default=None)

        if player_lowest_trump and (not opponent_lowest_trump or player_lowest_trump < opponent_lowest_trump):
            return 'player'
        else:
            return 'opponent'

    def step(self, action):
        """
        Step function to process an action and transition the environment.
        - Attack: Play a card (if valid)
        - Defend: Counter with a higher card or trump
        - Pass/Take: End the attack or take cards (defender)
        """
        if self.done:
            return self._get_obs(), 0, True, {}

        if self.player_turn == 'player':
            # Player's turn to attack or defend
            if action == 6:
                # Special action for "pass" or "take"
                if self.board:  # If defending and no valid defense, "take"
                    self._take_cards()
                else:
                    self._pass_turn()
            else:
                # Play the selected card if it's a valid move
                self._play_card(self.player_hand, action)
                self.player_turn = 'opponent'  # Change turn after action
        else:
            # Simulate opponent's action (random or opponent AI logic)
            opponent_action = self._opponent_action()
            if opponent_action == 6:
                if self.board:
                    self._take_cards()
                else:
                    self._pass_turn()
            else:
                self._play_card(self.opponent_hand, opponent_action)
                self.player_turn = 'player'

        reward = self._calculate_reward()
        self.done = self._check_done()
        return self._get_obs(), reward, self.done, {}

    def _play_card(self, hand, action):
        """Play a card from the hand to the board if valid."""
        card = hand[action]
        self.board.append(card)
        hand.remove(card)

    def _take_cards(self):
        """Take all cards from the board (when unable to defend)."""
        # Add all cards on board to the defender's hand
        if self.player_turn == 'player':
            self.player_hand.extend(self.board)
        else:
            self.opponent_hand.extend(self.board)
        self.board = []

    def _pass_turn(self):
        """End the attack and move to the next round."""
        self.board = []
        self._draw_cards()

    def _draw_cards(self):
        """Both players draw cards from the deck to refill hands to 6 cards."""
        while len(self.player_hand) < 6 and self.deck:
            self.player_hand.append(self.deck.pop(0))
        while len(self.opponent_hand) < 6 and self.deck:
            self.opponent_hand.append(self.deck.pop(0))

    def _opponent_action(self):
        """Random or AI logic for opponent's move (here using random as a placeholder)."""
        # In a more advanced implementation, this could be a trained model or heuristic
        if not self.board:  # If opponent is attacking
            return random.choice(range(len(self.opponent_hand)))
        else:
            return random.choice([i for i, card in enumerate(self.opponent_hand)])

    def _calculate_reward(self):
        """Reward function based on game outcome."""
        # In a more advanced implementation, each action (attack/defence) can do reward based on profit of the action
        if not self.opponent_hand:
            return 1  # Win
        if not self.player_hand:
            return -1  # Loss
        return 0  # Ongoing game, no immediate reward

    def _check_done(self):
        """Check if the game is over."""
        return not self.player_hand or not self.opponent_hand

    def _get_obs(self):
        """Return the current observation of the game."""
        return {
            "player_hand": self._get_hand_obs(self.player_hand),
            "opponent_hand_size": len(self.opponent_hand),
            "board_state": self.board[-2:] if len(self.board) >= 2 else [None, None],
            "trump_suit": self._suit_to_idx(self.trump_suit),
        }

    def _get_hand_obs(self, hand):
        """Convert player's hand to observation (integer indices for cards)."""
        return [self._card_to_idx(card) for card in hand]

    def _card_to_idx(self, card):
        """Convert a card (rank, suit) to an index for observation."""
        ranks = [6, 7, 8, 9, 10, 11, 12, 13, 14]
        suits = ['hearts', 'diamonds', 'clubs', 'spades']
        rank_idx = ranks.index(card[0])
        suit_idx = suits.index(card[1])
        return rank_idx * 4 + suit_idx

    def _suit_to_idx(self, suit):
        """Convert suit to an index."""
        suits = ['hearts', 'diamonds', 'clubs', 'spades']
        return suits.index(suit)

    def reset(self, seed=None, options=None):
        """Reset the game state."""
        self._initialize_game()
        self.done = False
        return self._get_obs(), {}
