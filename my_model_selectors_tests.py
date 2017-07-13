import unittest
import numpy as np
import pandas as pd
from asl_data import AslDb
import my_model_selectors
import importlib
importlib.reload(my_model_selectors)
import timeit


class MyDICTest(unittest.TestCase):

    def setUp(self):
        self.asl = AslDb()  # initializes the database
        self.asl.df.head()  # displays the first five rows of the asl database, indexed by video and frame
        self.asl.df['grnd-ry'] = self.asl.df['right-y'] - self.asl.df['nose-y']
        self.asl.df['grnd-rx'] = self.asl.df['right-x'] - self.asl.df['nose-x']
        self.asl.df['grnd-ly'] = self.asl.df['left-y'] - self.asl.df['nose-y']
        self.asl.df['grnd-lx'] = self.asl.df['left-x'] - self.asl.df['nose-x']
        self.features_ground = ['grnd-rx', 'grnd-ry', 'grnd-lx', 'grnd-ly']
        self.words_to_train = ['FISH', 'BOOK', 'VEGETABLE', 'FUTURE', 'JOHN']
        self.training = self.asl.build_training(self.features_ground)  # Experiment here with different feature sets defined in part 1

    def test_something(self):
        sequences = self.training.get_all_sequences()
        Xlengths = self.training.get_all_Xlengths()
        for word in self.words_to_train:
            start = timeit.default_timer()
            model = my_model_selectors.SelectorDIC(sequences, Xlengths, word,
                                                   min_n_components=2, max_n_components=15, random_state=14).select()
            end = timeit.default_timer() - start
            if model is not None:
                print("Training complete for {} with {} states with time {} seconds".format(word, model.n_components,
                                                                                            end))
            else:
                print("Training failed for {}".format(word))


if __name__ == '__main__':
    unittest.main()
