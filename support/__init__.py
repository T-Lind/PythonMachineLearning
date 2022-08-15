"""
Path to datasets
"""
DATASETS = "C:\\Users\\zenith\\Documents\\MyDatasets\\"


def display_scores(scores):
    """
    Display the scores from an evaluation
    """
    print("Scores: ", scores)
    print("Mean:", scores.mean())
    print("Standard deviation", scores.std())
