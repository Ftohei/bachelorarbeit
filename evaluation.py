from  gensim.models import Word2Vec
import numpy as np
import scipy.spatial.distance as scp
import file_util
import sys
import composition_learning
import matplotlib.pyplot as plt


################
#
#   Allgemein: in allen Trainings und Test-sets sind die Attribute großgeschrieben
#   für den embedding-space müssen sie allerdings klein geschrieben werden
################

HEIPLAS_DEV_SET_PATH = 'data/HeiPLAS-dev.txt'
HEIPLAS_TEST_SET_PATH = 'data/HeiPLAS-test.txt'
W2V_BINARY_PATH = '/Users/Fabian/Documents/Uni/6. Semester/Bachelorarbeit/code/GoogleNews-vectors-negative300.bin'

def test_for_double_entries(list):
    names = [name for name,cossim,isbla in list]
    counts = [(elem,names.count(elem)) for elem in names]
    result = []
    for i in counts:
        if i[1] != 1 and i not in result:
            result.append(i)
    return result

def compute_mitchell_lapata(u,v, factor, verbosity = 0):
    #implementiert  p = (u * u) * v + (factor - 1) * (u * v) * u (Mitchell Lapata 2010)
    if verbosity >= 2: print("Shapes von u und v in compute_mitchell_lapata: {} und {}\tFactor = {}".format(np.shape(u),np.shape(v),factor))
    dot_uu = np.dot(u,u)
    dot_uv = np.dot(u,v)
    result = np.add(np.multiply(dot_uu, v) , np.multiply(factor - 1, np.multiply(dot_uv, u)))
    if verbosity >= 2: print("Shape von result in compute_mitchell_lapata; {}".format(np.shape(result)))
    return result

def compute_vector_mean(a,b, verbosity = 2):
    if verbosity >= 2 and np.shape(a) != np.shape(b):
        print("Ungleiche Dimensionen bei Mittelwertbildung!")
        return np.zeros(np.shape(a))
    else:
        result = np.zeros(np.shape(a))
        for i in range(0, np.shape(a)[0]):
            result[i] = np.mean([a[i],b[i]])
        return result

def compute_vector_max(a,b,verbosity=2):
    if verbosity >= 2 and np.shape(a) != np.shape(b):
        print("Ungleiche Dimensionen bei Maxwertbildung!")
        return np.zeros(np.shape(a))
    else:
        result = np.zeros(np.shape(a))
        for i in range(0, np.shape(a)[0]):
            result[i] = float(np.max([a[i],b[i]]))
        return result

def cos_sim(vec1,vec2, verbosity = 0):
    if not (np.shape(vec1) == np.shape(vec2)):
        print("Ungleiche Dims bei Cos Sim!")
    else:
        # try:
        return 1 - scp.cosine(vec1,vec2)
        # except:


def find_next_neighbours(adj, noun, attr, vectorspace, models, attr_test_set, top_n = 10, projection_mode = 'add', verbosity = 2):
    #suche für die adj-nomen-phrase mit einem bestimmten Projektions-Modus
    # die top_n nächsten Attribute aus einem attr_test_set
    result_cosine_sim = []
    projected_vec = np.ndarray((0,))
    word_found = True
    not_in_embedding_space = []

    try:
        if projection_mode == 'adj':
            projected_vec = vectorspace[adj]
        elif projection_mode == 'noun':
            projected_vec = vectorspace[noun]
        elif projection_mode == 'add':
            projected_vec = np.add(vectorspace[adj], vectorspace[noun])
        elif projection_mode == 'mult':
            projected_vec = np.multiply(vectorspace[adj], vectorspace[noun])
        elif projection_mode == 'sub_a-n':
            projected_vec = np.subtract(vectorspace[adj], vectorspace[noun])
        elif projection_mode == 'sub_n-a':
            projected_vec = np.subtract(vectorspace[noun], vectorspace[adj])
        elif projection_mode == 'avg':
            projected_vec = compute_vector_mean(vectorspace[adj],vectorspace[noun],verbosity=verbosity)
        elif projection_mode == 'max':
            projected_vec = compute_vector_max(vectorspace[adj],vectorspace[noun],verbosity=verbosity)
        elif projection_mode == 'mitchell_lapata':
            projected_vec = compute_mitchell_lapata(vectorspace[adj], vectorspace[noun], 2)
        elif projection_mode == 'mitchell_lapata_reversed':
            projected_vec = compute_mitchell_lapata(vectorspace[noun], vectorspace[adj], 2)
        elif projection_mode == 'nn_tensor_product_random':
            projected_vec = models['nn_tensor_product_random'].predict(np.asarray([[vectorspace[adj], vectorspace[noun]]]))[0, 0]
        elif projection_mode == 'nn_tensor_product_identity':
            projected_vec = models['nn_tensor_product_identity'].predict(np.asarray([[vectorspace[adj], vectorspace[noun]]]))[0, 0]
        else:
            projected_vec = models[projection_mode].predict(np.asarray([[vectorspace[adj], vectorspace[noun]]]))[0]

    except KeyError as e:
        if verbosity>=2:
            print("Adjektiv oder Nomen beim Test nicht in WordEmbeddings enthalten: {},{}".format(adj,noun))
        word_found = False
        if adj not in not_in_embedding_space and noun not in not_in_embedding_space:
            not_in_embedding_space += [adj,noun]


    if word_found:
        for test_attr in [test_attr.lower() for test_attr in attr_test_set]:
            # print word
            try:
                projected_attr = vectorspace[test_attr.lower()]
                cosine_sim = cos_sim(projected_vec, projected_attr)
                if test_attr == attr:    #hier wird noch markiert, ob es sich um das gesuchte attribut handelt!
                    result_cosine_sim.append((test_attr,cosine_sim,True))
                else:
                    result_cosine_sim.append((test_attr,cosine_sim,False))
            except KeyError:
                if verbosity >= 2:
                    print("Attribut beim Test nicht in WordEmbeddings enthalten: {}".format(test_attr))
                if test_attr not in not_in_embedding_space:
                    not_in_embedding_space += [test_attr]

    #sortiere nach der kosinus-ähnlichkeit absteigend.
    result_cosine_sim.sort(key = lambda x:x[1], reverse=True)

    return result_cosine_sim[0:top_n], not_in_embedding_space

def train_models(attr_train_set, vectorspace, proj_mode_list, verbosity = 2):
    aan_list_dev, attrs_dev, adjs_dev, nouns_dev = file_util.read_attr_adj_noun(HEIPLAS_DEV_SET_PATH, vectorspace=vectorspace,
                                                                                verbosity=1)

    if verbosity >= 2:
        print("Alle AANs aus dev ({}): ".format(len(aan_list_dev)), aan_list_dev)
        print("Trainings-Menge: ", attr_train_set)


    data, labels, not_in_embedding_space = composition_learning.construct_data_and_labels(aan_list_dev,
                                                                                          vectorspace,
                                                                                          attr_train_set=attr_train_set,
                                                                                          verbosity=verbosity)
    if verbosity >= 1:
        print("Beim Training nicht im Embedding-Raum: {}".format(not_in_embedding_space))
    if verbosity >= 2:

        print("Data/Label-Shape:",np.shape(data),np.shape(labels))
    models = {}

    #werden in bestimmter reihenfolge abgearbeitet
    if 'nn_tensor_product_random' in proj_mode_list:
        nn_model_tensor_prod = composition_learning.train_model(data, labels, composition_mode='tensor_mult_random', verbosity=verbosity)
        models['nn_tensor_product_random'] = nn_model_tensor_prod
    if 'nn_tensor_product_identity' in proj_mode_list:
        nn_model_tensor_prod_identity = composition_learning.train_model(data, labels, composition_mode='tensor_mult_identity', verbosity=verbosity)
        models['nn_tensor_product_identity'] = nn_model_tensor_prod_identity
    if 'nn_weighted_adjective_identity' in proj_mode_list:
        nn_model_w_adj_identity = composition_learning.train_model(data, labels, composition_mode='weighted_adj_add_identity', verbosity=verbosity)
        models['nn_weighted_adjective_identity'] = nn_model_w_adj_identity
    if 'nn_weighted_noun_identity'  in proj_mode_list:
        nn_model_w_noun_identity = composition_learning.train_model(data, labels, composition_mode='weighted_noun_add_identity', verbosity=verbosity)
        models['nn_weighted_noun_identity'] = nn_model_w_noun_identity
    if 'nn_weighted_adjective_noun_identity'  in proj_mode_list:
        nn_model_w_adj_noun_identity = composition_learning.train_model(data, labels, composition_mode='weighted_adj_and_noun_add_identity', verbosity=verbosity)
        models['nn_weighted_adjective_noun_identity'] = nn_model_w_adj_noun_identity
    if 'nn_weighted_adjective_random'  in proj_mode_list:
        nn_model_w_adj_random = composition_learning.train_model(data, labels, composition_mode='weighted_adj_add_random', verbosity=verbosity)
        models['nn_weighted_adjective_random'] = nn_model_w_adj_random
    if 'nn_weighted_noun_random' in proj_mode_list:
        nn_model_w_noun_random = composition_learning.train_model(data, labels, composition_mode='weighted_noun_add_random', verbosity=verbosity)
        models['nn_weighted_noun_random'] = nn_model_w_noun_random
    if 'nn_weighted_adjective_noun_random' in proj_mode_list:
        nn_model_w_adj_noun_random = composition_learning.train_model(data, labels, composition_mode='weighted_adj_and_noun_add_random', verbosity=verbosity)
        models['nn_weighted_adjective_noun_random'] = nn_model_w_adj_noun_random
    if 'nn_weighted_adjective_ones' in proj_mode_list:
        nn_model_w_adj_ones = composition_learning.train_model(data, labels, composition_mode='weighted_adj_add_ones', verbosity=verbosity)
        models['nn_weighted_adjective_ones'] = nn_model_w_adj_ones
    if 'nn_weighted_noun_ones'  in proj_mode_list:
        nn_model_w_noun_ones = composition_learning.train_model(data, labels, composition_mode='weighted_noun_add_ones', verbosity=verbosity)
        models['nn_weighted_noun_ones'] = nn_model_w_noun_ones
    if 'nn_weighted_adjective_noun_ones' in proj_mode_list:
        nn_model_w_adj_noun_ones = composition_learning.train_model(data, labels, composition_mode='weighted_adj_and_noun_add_ones', verbosity=verbosity)
        models['nn_weighted_adjective_noun_ones'] = nn_model_w_adj_noun_ones
    if 'nn_weighted_adj_and_noun_add_identity_with_rands' in proj_mode_list:
        nn_model_w_adj_noun_identity_with_rands = composition_learning.train_model(data, labels, composition_mode='weighted_adj_and_noun_add_identity_with_rands', verbosity=verbosity)
        models['nn_weighted_adj_and_noun_add_identity_with_rands'] = nn_model_w_adj_noun_identity_with_rands
    if 'nn_weighted_adj_noun_add_sum1_identity' in proj_mode_list:
        nn_model_w_adj_noun_identity_with_rands = composition_learning.train_model(data, labels, composition_mode='weighted_adj_noun_add_sum1_identity', verbosity=verbosity)
        models['nn_weighted_adj_noun_add_sum1_identity'] = nn_model_w_adj_noun_identity_with_rands
    if 'nn_weighted_adj_noun_add_sum1_random' in proj_mode_list:
        nn_model_w_adj_noun_identity_with_rands = composition_learning.train_model(data, labels, composition_mode='weighted_adj_noun_add_sum1_random', verbosity=verbosity)
        models['nn_weighted_adj_noun_add_sum1_random'] = nn_model_w_adj_noun_identity_with_rands
    return models

def compute_tables(complete_results, proj_mode_list):
    head_line = ""
    for proj_mode in proj_mode_list:
        head_line += "{:<40}".format(proj_mode.upper())

    for adj,noun,attr,results_for_aan in complete_results:    #über alle aans
        print("\n{} x {} -> {}".format(adj,noun,attr))
        print(head_line)

        num_rows = len(results_for_aan[0][1])
        for i in range(0,num_rows):#über alle ergebnisse der proj_mode_listen ('RÄNGE') -> entspricht je einer ZEILE der Tabelle
            string = ""
            num_cols = len(proj_mode_list)
            for j in range(0, num_cols):  #über alle proj_modes, entspricht je einer SPALTE der Tabelle
                projection_mode, top_n_results = results_for_aan[j]         #je immer die resultate für Proj_mode j (spalte j
                top_attr,cos_similarity,is_searched_attr = top_n_results[i]     # die resultate für Rang i (zeile
                double_entries = test_for_double_entries(top_n_results)
                if double_entries:
                    print("Doppelte Einträge! {}".format(double_entries))
                if top_attr == attr:
                    top_attr = top_attr.upper()
                else:
                    top_attr = top_attr.lower()
                string += "{:<5}{:<25}{:<10.2f}".format(i+1,top_attr,cos_similarity)    #fügt einen Spalten-Eintrag in die Zeile ein: Nummer, Attribut, CosSim
            print(string)       # printe fertige Zeile

def compute_quantitive_eval(complete_results, proj_mode_list, verbosity = 2, sort_list = 2):
    #sort_list = 1 ist sortierung nach prec@1, = 2 nach prec@5, = 0 keine umsortierung

    max_len_proj_mode = np.max([len(string) for string in proj_mode_list])

    counters = {}   #dict mit projection_mode:[insgesamt,prec@1,prec@5]

    for proj_mode in proj_mode_list:
        counters[proj_mode] = [0,0,0]

    for adj,noun,attr,results_for_aan in complete_results:
        for proj_mode, results_for_proj_mode in results_for_aan:
            counters[proj_mode][0] += 1

            if results_for_proj_mode:
                attr,cos_similarity,is_searched_attr = results_for_proj_mode[0]
                if is_searched_attr:
                    counters[proj_mode][1] += 1

                in_top_5 = False
                for attr,cos_similarity,is_searched_attr in results_for_proj_mode[0:5]:
                    if is_searched_attr:
                        in_top_5 = True

                if in_top_5:
                    counters[proj_mode][2] += 1
            else:
                if verbosity >= 2:
                    print("Keine Resultate für {},{} -> {}".format(adj,noun,attr))



    header = "{:<{}}{:<10}{:<10}".format(
        '',
        max_len_proj_mode + 5,
        'Prec@1',
        'Prec@5'
    )
    print(header)

    results = []
    for proj_mode in proj_mode_list:
        prec_at_1 = float(counters[proj_mode][1])/counters[proj_mode][0]
        prec_at_5 = float(counters[proj_mode][2])/counters[proj_mode][0]
        results.append((proj_mode,prec_at_1,prec_at_5))

    if sort_list != 0:
        results = sorted(results, key = lambda x : x[sort_list], reverse=True)

    for result in results:
        proj_mode_string= "{:<{}}{:<10.2f}{:<10.2f}".format(
            result[0],
            max_len_proj_mode + 5,
            result[1],
            result[2]
        )
        print(proj_mode_string)

def plot_hist(complete_results, proj_mode_list, verbosity = 2, sort_list = 2):
    #TODO print beautiful plots of results
    counters = {}   #dict mit projection_mode:[insgesamt,prec@1,prec@5]

    for proj_mode in proj_mode_list:
        counters[proj_mode] = [0,0,0]

    for adj,noun,attr,results_for_aan in complete_results:
        for proj_mode, results_for_proj_mode in results_for_aan:
            counters[proj_mode][0] += 1

            if results_for_proj_mode:
                attr,cos_similarity,is_searched_attr = results_for_proj_mode[0]
                if is_searched_attr:
                    counters[proj_mode][1] += 1

                in_top_5 = False
                for attr,cos_similarity,is_searched_attr in results_for_proj_mode[0:5]:
                    if is_searched_attr:
                        in_top_5 = True

                if in_top_5:
                    counters[proj_mode][2] += 1
            else:
                if verbosity >= 2:
                    print("Keine Resultate für {},{} -> {}".format(adj,noun,attr))

    prec_at_1_list = []
    prec_at_5_list = []
    labels = list(counters)
    for proj_mode in labels:
        prec_at_1_list.append(counters[proj_mode][1])
        prec_at_5_list.append(counters[proj_mode][2])

    plt.hist(prec_at_1_list, facecolor='green', alpha=0.5)
    plt.hist(prec_at_5_list, facecolor='blue', alpha=0.5)
    plt.xticks(prec_at_1_list, labels, rotation='vertical')
    plt.margins(0.2)
    plt.show()

def evaluate(attr_train_set, attr_test_set, proj_mode_list, tables=False, quantitive_eval=True, plot=True, train_test_exclusivity=False, verbosity=2):
    if verbosity >= 1:
        print("Lade word-embeddings...")
    vectorspace = Word2Vec.load_word2vec_format(W2V_BINARY_PATH, binary=True)
    if verbosity >= 1:
        print("word-embeddings fertig geladen")

    sys.stdout.flush()

    aan_list_test, attrs_test, adjs_test, nouns_test = file_util.read_attr_adj_noun(HEIPLAS_TEST_SET_PATH, vectorspace=vectorspace, verbosity=1)

    if train_test_exclusivity:
        attr_test_set = [attr for attr in attr_test_set if attr not in attr_train_set]

    if verbosity >= 1:
        print('Training auf: %s\nTest auf %s' % (attr_train_set,attr_test_set))
    models = train_models(attr_train_set, vectorspace, proj_mode_list, verbosity=verbosity)

    complete_results = [] #liste mit results_for_aan

    not_in_space = []

    for aan in [aan for aan in aan_list_test if aan[0].upper() in attr_test_set]: #nur die aus dem Test-set!
        results_for_aan = []
        attr = aan[0]
        adj = aan[1]
        noun = aan[2]

        for projection_mode in proj_mode_list:

            top_n_results, not_in_embedding_space = find_next_neighbours(adj,noun,attr,vectorspace,models,attr_test_set,projection_mode=projection_mode, verbosity=verbosity)

            #HIER könnte man testen ob ergebnisse doppelt vorkommen!
            if verbosity >=1:
                counts = [top_n_results.count(elem) for elem in top_n_results]
                if counts.count(1) < len(counts):
                    print("find_next_neighbours gibt für dasselbe aan ergebnisse doppelt zurück! {}".format(top_n_results))

            # print(projection_mode,top_n_results)
            results_for_aan.append((projection_mode,top_n_results)) #tupel : (projection_mode, liste der top_n resultate für den aktuellen proejction mode)
            #reminder: ein top_result ist ein Tripel mit (attribut,cos_sim,BINäR: ist es das gesucht attribut?)
            not_in_space += [i for i in not_in_embedding_space if i not in not_in_space]

        complete_results.append((adj,noun,attr,results_for_aan))

    if verbosity>=1:
        print("Beim Test nicht im Embedding-Raum: {}".format(not_in_space))

        sys.stdout.flush()

    if quantitive_eval:
        compute_quantitive_eval(complete_results, proj_mode_list, verbosity=verbosity)
    if tables:
        compute_tables(complete_results, proj_mode_list)
    if plot:
        plot_hist(complete_results, proj_mode_list)
