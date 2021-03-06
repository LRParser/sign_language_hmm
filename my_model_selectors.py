import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN

    """

    # NOTE: L is the likelihood of the fitted model, p is the number of parameters, and N is the number of data points

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        best_model = None
        best_num_components = 0

        # The
        best_score = float("inf")

        for i in range(self.min_n_components,self.max_n_components + 1) :

            if i >= self.lengths[0] :
                continue

            try :

                hmm_model = GaussianHMM(n_components=i, covariance_type="diag", n_iter=1000,
                                        random_state=self.random_state, verbose=False).fit(self.X, self.lengths)

                logL = hmm_model.score(self.X, self.lengths)
                N = len(self.lengths)
                p = i * i + 2 * i * len(self.X[0]) - 1
                bic_score = -2 * logL + p * math.log(N)

            except Exception as e:
                # print(e)
                return None

            # print("score of {} for model with {} components".format(bic_score, i))

            if bic_score < best_score :
                best_score = bic_score
                best_model = hmm_model
                best_num_components = i

        # print("Best score of {} for model with {} components".format(best_score,best_num_components))
        return best_model


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on DIC scores
        # Question to reviewer - Should we look at discriminitive capacity of some number of states vs other number of states, or of self.this_word vs other words? I think the former but want to clarify
        best_model = None
        best_num_components = 0

        # The
        best_score = float("-inf")

        # print("Iterate thru range")
        for i in range(self.min_n_components, self.max_n_components + 1):

            # print("Testing n_components at: {}".format(i))

            try:

                if i > self.lengths[0]:
                    continue

                hmm_model = GaussianHMM(n_components=i, covariance_type="diag", n_iter=1000,
                                        random_state=self.random_state, verbose=False).fit(self.X, self.lengths)

                # Question to reviewer - it is a bug in HMMLearn that this sometimes returns a negative number as a log probability
                # https://discussions.udacity.com/t/logl-negative-in-cell-with-chocolate-word/231882/4
                this_model_score = hmm_model.score(self.X, self.lengths)
                other_model_scores = 0

                m_count = 0
                for word in self.words :
                    if word == self.this_word :
                        continue
                    else :
                        # print("Compare base word {} with {}".format(self.this_word,word))

                        X, lengths = self.hwords[word]

                        if i > lengths[0]:
                            continue

                        other_hmm_model = GaussianHMM(n_components=i, covariance_type="diag", n_iter=1000,
                                                      random_state=self.random_state, verbose=False).fit(X,
                                                                                                         lengths)

                        try:
                            other_hmm_model_score = other_hmm_model.score(X, lengths)
                        except Exception as e:
                            # print(e)
                            # print("Scoring of comparative model failed, base word {}, compare word {}".format(self.this_word,word))
                            # Question to the reviewer - should I penalize an HmmLearn failure by setting other_hmm_model_score to float(-inf) or is 0 more appropriate?
                            # Example error: rows of transmat_ must sum to 1.0 (got [ 1.  1.  1.  0.  1.  1.  0.  1.  1.])
                            other_hmm_model_score = 0

                        other_model_scores += other_hmm_model_score
                        m_count += 1

                        # DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))

                    dic_score = this_model_score - (1 / m_count) * other_model_scores
                    # print("dic_score for {} is {}".format(i,dic_score))

                    if dic_score > best_score:
                        best_score = dic_score
                        best_model = hmm_model
                        best_num_components = i

            except Exception as e:
                continue
                # print("Error for i = {} and word = {}".format(i,self.this_word))
                # print(e)

        # print("Best score of {} for model with {} components".format(best_score, best_num_components))
        return best_model


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        word_sequences = self.sequences
        num_splits = self.min_n_components

        if(len(word_sequences) < num_splits) :
            return None

        split_method = KFold(n_splits=num_splits)
        best_model = None
        best_num_components = 0

        # The
        best_score = float("-inf")

        for i in range(self.min_n_components,self.max_n_components) :

            total_score = 0


            for cv_train_idx, cv_test_idx in split_method.split(word_sequences):
                # print("Train fold indices:{} Test fold indices:{}".format(cv_train_idx, cv_test_idx))  # view indices of the folds

                X, X_lengths = combine_sequences(cv_train_idx,word_sequences)
                Y, Y_lengths = combine_sequences(cv_test_idx,word_sequences)

                try :

                    hmm_model = GaussianHMM(n_components=i, covariance_type="diag", n_iter=1000,
                                            random_state=self.random_state, verbose=False).fit(X, X_lengths)

                    logL = hmm_model.score(Y, Y_lengths)
                    total_score = total_score + logL

                except Exception as e :
                    # print(e)
                    continue

            avg_score = total_score / num_splits
            # print("avg_score of {} for model with {} components".format(avg_score, i))

            if avg_score > best_score :
                best_score = avg_score
                best_model = hmm_model
                best_num_components = i

        # print("Best score of {} for model with {} components".format(best_score,best_num_components))

        return best_model
