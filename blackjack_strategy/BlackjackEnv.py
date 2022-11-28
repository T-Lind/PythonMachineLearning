import random


class Deck:
    def __init__(self):
        self.deck = [10] * 16 + [x for x in range(1, 10)] * 4
        random.shuffle(self.deck)

    def draw(self) -> int:
        itemOne = self.deck[0]
        del self.deck[0]
        return itemOne

    def is_empty(self) -> bool:
        return len(self.deck) == 0


class BlackjackEnv:
    def __init__(self, n_players, n_decks=1):
        self.cards = Deck()

        self.hands = [self.cards.draw() + self.cards.draw() for _ in range(n_players)] * n_decks

    def check_over(self, hand) -> bool:
        return self.hands[hand] > 21

    def add_card(self, hand) -> None:
        if not self.cards.is_empty():
            self.hands[hand] += self.cards.draw()

    def __str__(self):
        ret_str = "Current game hands:\n"
        for i in range(len(self.hands)):
            ret_str += f"Player {i} has a value of {self.hands[i]} and is{' not' if self.check_over(i) else ''} in the game\n"
        return ret_str


class Actor:
    def __init__(self, game: BlackjackEnv, index: int):
        self.game = game
        self.index = index

    def hit_turn(self):
        self.game.add_card(self.index)

    def pass_turn(self):
        pass

    def get_value(self):
        return self.game.hands[self.index]
