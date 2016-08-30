import numpy as np
from gensim.models import Word2Vec
import file_util
from scipy.stats import spearmanr
import composition_learning
from evaluation import *
from subset_experiments import *


def compute_similarities(ratings, vector_space, models, proj_mode ='add', verbosity=0):
    """
    Copmutes similarities for phrase pairs.
    :param ratings: phrase pairs with human ratings.
    :param vector_space: pre-trained word embeddings.
    :param models: a list containing trained neural models.
    :param proj_mode: specifies a certain projection mode.
    :param verbosity:
    :return:
    """

    result = []
    num_ratings = np.shape(ratings)[0]

    for i in range(0,num_ratings):
        phrasevec_1 = []
        phrasevec_2 = []

        adj1 = ratings[i,0][0]
        noun1 = ratings[i,0][1]
        adj2 = ratings[i,1][0]
        noun2 = ratings[i,1][1]

        success = True

        michtell_lapata_dilation_factor = 2

        try:
            if proj_mode == 'add':
                phrasevec_1 = vector_space[adj1] + vector_space[noun1]
            elif proj_mode == 'mult':
                phrasevec_1 = np.multiply(vector_space[adj1], vector_space[noun1])
            elif proj_mode == 'mitchell_lapata_2':
                phrasevec_1 = compute_mitchell_lapata(vector_space[adj1], vector_space[noun1], michtell_lapata_dilation_factor, verbosity=verbosity)
            elif proj_mode == 'mitchell_lapata_reversed_2':
                phrasevec_1 = compute_mitchell_lapata(vector_space[noun1], vector_space[adj2], michtell_lapata_dilation_factor, verbosity=verbosity)
            else:
                phrasevec_1 = models[proj_mode].predict(np.asarray([[vector_space[adj1], vector_space[noun1]]]))[0]
        except KeyError:
            print(proj_mode)
            print("%s oder %s nicht im vektorspace enthalten" % (adj1, noun1))
            success = False

        try:
            if proj_mode == 'add':
                phrasevec_2 = vector_space[adj2] + vector_space[noun2]
            elif proj_mode == 'mult':
                phrasevec_2 = np.multiply(vector_space[adj2], vector_space[noun2])
            elif proj_mode == 'mitchell_lapata_2':
                phrasevec_2 = compute_mitchell_lapata(vector_space[adj2], vector_space[noun2], michtell_lapata_dilation_factor, verbosity=verbosity)
            elif proj_mode == 'mitchell_lapata_reversed_2':
                phrasevec_2 = compute_mitchell_lapata(vector_space[noun2], vector_space[adj2], michtell_lapata_dilation_factor, verbosity=verbosity)
            else:
                phrasevec_2 = models[proj_mode].predict(np.asarray([[vector_space[adj2], vector_space[noun2]]]))[0]
        except KeyError:
            print("%s oder %s nicht im vektorspace enthalten" % (adj2, noun2))
            success = False

        cosine_similarity = 0
        if success:
            if verbosity >= 2: print(np.shape(phrasevec_1), np.shape(phrasevec_2))
            cosine_similarity = cos_sim(phrasevec_1, phrasevec_2)

        result.append(cosine_similarity)

    return np.asarray(result).T

SIMILARITY_RATINGS_PATH = "data/mitchell-lapata-sim-ratings.txt"

def fetch_similarity_ratings(path, ordered = False, vectorspace=False, verbosity = 0):
    """Reads phrase pairs and similarity ratings from a file and returns them in descending order by human ratings"""
    sim_ratings = file_util.read_sim_ratings(path, vectorspace=vectorspace, verbosity=verbosity)
    if ordered:
        ordered_ratings = []
        for sim_rating in sim_ratings:
            ordered_ratings.append([(sim_rating[0],sim_rating[1]),
                                    (sim_rating[2],sim_rating[3]),
                                    int(sim_rating[4])])

        sim_ratings = sorted(ordered_ratings, key = lambda x:x[2], reverse=True)

    return np.asarray(sim_ratings)

def spearman_evaluation(similarity_tuple_list, verbosity = 2):
    """Computes spearman's r between human ratings and different projection modes."""
    print("-------------Similarity-Ratings: Spearman's r-------------")
    longest_name = np.max([len(mode) for mode,ratings in similarity_tuple_list])

    header = "{:<{}}{:<15}{:<15}".format('', longest_name + longest_name + 10, 'Correlation','P-Value')
    print(header)
    for mode, similarities in similarity_tuple_list:
        print()
        if 'human' in mode:
            for mode2, similarities2 in similarity_tuple_list:
                if mode != mode2:
                    if verbosity >= 2: print("Similarities bei {}: {}\nSimilarities bei {}: {}".format(mode,similarities,mode2,similarities2))
                    spearman = spearmanr(similarities, similarities2)
                    corr = spearman[0]
                    p = spearman[1]
                    print("{:<{}}{:<15.3f}{:<15.6f}".format(mode + ' vs ' + mode2,longest_name + longest_name + 10,corr,p))

def source_of_similarity_eval(ratings, attr_test_set, vectorspace, models, proj_mode, top_n=5, verbosity=2):
    """Prints tables containing the top attributes for given phrase pairs. Used for evaluation of attributes as the source of similarity"""
    ratings = ratings[:,:2]
    for (adj1,noun1), (adj2,noun2) in ratings.tolist():

        top_n_results1, not_in_embedding_space1 = find_next_neighbours(adj1, noun1, '', vectorspace,
                                                                       models,attr_test_set, top_n=top_n, projection_mode=proj_mode,
                                                                       verbosity=verbosity)

        top_n_results2, not_in_embedding_space2 = find_next_neighbours(adj2, noun2, '', vectorspace,
                                                                       models,attr_test_set, top_n=top_n, projection_mode=proj_mode,
                                                                       verbosity=verbosity)

        shared_attrs = [test_attr for test_attr,cosine,is_searched_attr in top_n_results1
                        if test_attr in [attr for attr,cos,isbla in top_n_results2]]

        phrase1 = adj1+"_"+noun1
        phrase2 = adj2+"_"+noun2
        header = "\n{:<30}{:^7}{:<30}({})".format(phrase1.upper(),':',phrase2.upper(),proj_mode)
        print(header)


        for (test_attr1,cosine1,boolean1),(test_attr2,cosine2,boolean2) in list(zip(top_n_results1,top_n_results2)):
            if test_attr1 in shared_attrs:
                test_attr1 = test_attr1.upper()
            else:
                test_attr1 = test_attr1.lower()
            if test_attr2 in shared_attrs:
                test_attr2 = test_attr2.upper()
            else:
                test_attr2 = test_attr2.lower()
            print("{:<25}{:>5.2f}{:^7}{:<25}{:>5.2f}".format(test_attr1, cosine1,':',test_attr2,cosine2))

def source_of_similarity_quantitative_eval(ratings, attr_test_set, vectorspace, models, proj_mode, top_n=5, verbosity=2):
    """Computes quantitative evaluation metrics for the evaluation of attributes as the source of similarity"""
    ratings = ratings[:,:2]
    phrase_pair_counter = 0
    prec_at_1_counter = 0
    source_of_sim_metric = 0

    for (adj1,noun1), (adj2,noun2) in ratings.tolist():
        top_n_results1, not_in_embedding_space1 = find_next_neighbours(adj1, noun1, '', vectorspace,
                                                                       models,attr_test_set, top_n=top_n, projection_mode=proj_mode,
                                                                       verbosity=verbosity)

        top_n_results2, not_in_embedding_space2 = find_next_neighbours(adj2, noun2, '', vectorspace,
                                                                       models,attr_test_set, top_n=top_n, projection_mode=proj_mode,
                                                                       verbosity=verbosity)

        phrase_pair_counter += 1

        if top_n_results1[0][0] == top_n_results2[0][0]:
            prec_at_1_counter += 1

        shared_results = [attr for attr,cosine,boolean in top_n_results1 if attr in [attr2 for attr2,cosine2,boolean2 in top_n_results2]]

        source_of_sim_metric += len(shared_results) / top_n

    prec_at_1 = prec_at_1_counter / phrase_pair_counter

    source_of_sim_metric /= phrase_pair_counter

    print("{:<45}{:<10.2f}{:<10.2f}".format(proj_mode,prec_at_1,source_of_sim_metric))


def similarity_experiment(attr_train_set, attr_test_set, proj_mode_list, spearman = True, tables = True, quantitative = True, rating_cutoff_list = [4,5], verbosity = 1):
    """Performs similarity experiments for a given attribute training set, a given attribute test set and a list of projection modes"""
    if verbosity >= 1:
        print("lade word2vec-model...")
        print("Nutze train-set: {}".format(attr_train_set))
        print("Nutze test-set: {}".format(attr_test_set))
    vector_space = Word2Vec.load_word2vec_format(W2V_BINARY_PATH, binary=True)

    models = train_models(attr_train_set[1], vectorspace=vector_space, proj_mode_list=proj_mode_list, verbosity=verbosity)

    ratings = fetch_similarity_ratings(SIMILARITY_RATINGS_PATH, ordered=True, vectorspace=vector_space, verbosity=1)

    similarities = [('human',ratings[:,2].T)] #menschliche Ratings

    for proj_mode in proj_mode_list:
        if verbosity >= 2: print("Projection-mode: {}".format(proj_mode))
        similarities.append((proj_mode,compute_similarities(ratings, vector_space, models, proj_mode=proj_mode, verbosity=verbosity))) #pro proj_mode eine Liste mit Kosinus-Ähnlichkeiten

    if spearman:
        spearman_evaluation(similarities, verbosity)

    if rating_cutoff_list:
        for rating_cutoff in rating_cutoff_list:
            ratings = np.array(
                [((adj1,noun1),(adj2,noun2),int(rating)) for (adj1,noun1),(adj2,noun2),rating in ratings.tolist() if int(rating) > rating_cutoff]
            )   #hier gibt es sicher einen sehr hübschen weg, aber da das nicht oft passiert einfach ein quickfix

            print("\n--------------Rating-Cutoff: Human Ratings > {}---------------\nNach Cut-Off werden {} verbleibende Phrasen-Paare angeschaut".format(rating_cutoff, len(ratings)))

            if quantitative:
                print("\n-------------Source of Similarity: Quantitative Evaluierung-------------")

                print("{:<45}{:<10}{:<10}".format('','prec@1','average-shared_attribs-top-5'))    #header
                for proj_mode in proj_mode_list:
                    source_of_similarity_quantitative_eval(ratings, attr_test_set=attr_test_set[1], vectorspace=vector_space, models=models, proj_mode=proj_mode, verbosity=0)

            if tables:
                print("\n-------------Source of Similarity: Tabellen-Ausgabe-------------")

                for proj_mode in proj_mode_list:
                    print("---------------------------------{}-----------------------------".format(proj_mode.upper()))
                    source_of_similarity_eval(ratings, attr_test_set=attr_test_set[1], vectorspace=vector_space, models=models, proj_mode=proj_mode, verbosity=0)