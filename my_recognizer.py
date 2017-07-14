import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []

    best_word = None

    all_lengths = test_set.get_all_Xlengths()
    for word_id in all_lengths.keys() :
        test_sequence = all_lengths[word_id]
        test_X = test_sequence[0]
        test_Xlength = test_sequence[1]
        test_word = test_set.wordlist[word_id]
        # print("Test word_id {} and word {}".format(word_id,test_word))
        prob_map = dict()
        max_prob = float("-inf")

        for word, model in models.items():

            try:
                score = model.score(test_X,test_Xlength)
            except Exception as e :
                # print(e)
                score = -1000

            prob_map[word] = score
            if score > max_prob :
                max_prob = score
                best_word = word

        probabilities.append(prob_map)
        guesses.append(best_word)

    return probabilities, guesses