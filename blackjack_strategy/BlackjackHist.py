from BlackjackEnv import Deck
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    n_iters = 1_000_000

    two_cards = []
    three_cards = []

    for _ in range(n_iters):
        deck = Deck()
        value = deck.draw() + deck.draw()
        two_cards.append(value)
        three_cards.append(value + deck.draw())

    plt.hist(two_cards, 20, color="orange", alpha=0.6, label="Two drawn cards sum")
    plt.hist(three_cards, 20, color="red", alpha=0.6, label="Three drawn cards sum")

    print(f"Two cards mean: {np.mean(two_cards)}, median: {np.median(two_cards)}, st. dev: {np.std(two_cards)}")
    print(f"Three cards mean: {np.mean(three_cards)}, median: {np.median(three_cards)}, st. dev: {np.std(three_cards)}")

    plt.legend()
    plt.show()
