from similarity_ratings_tests import *
from subset_experiments import *
import sys
import os
import datetime

now = datetime.datetime.now()


subset_test_explanation = "1. Experiment, normale Version:\nHier werden je verschiedene Kompositionen auf verschiedenen Teilmengen der Attribute trainiert und getestet." \
                          " Dabei sind Trainings- und Testmenge nicht zwingend exklusiv. Nimmt man beispielsweise All als Trainingsmenge," \
                          " dann sind alle anderen Mengen darin enthalten.\nPrec@1 heißt, dass das Target-Attribut im Embedding-Raum nach " \
                          " Kosinus-Ähnlichkeit das ähnlichste Attribut der proijizierten Adjektiv-Nomen-Komposition ist. Prec@5 bedeutet, dass" \
                          " das Attribut noch eines der nächsten 5 Attribute ist.\nTraining auf dem HeiPLAS-dev-set, Test auf dem dem HeiPLAS-test-set."

subset_test_exclusive_explanation = "1. Experiment, zero-shot Version:\nHier werden je verschiedene Kompositionen auf verschiedenen Teilmengen der Attribute trainiert und getestet." \
                          " Hier sind Trainings- und Testmenge zwingend exklusiv. Alle Attribute aus der Trainingsmenge werden explizit aus der" \
                            " Testmenge entfernt.\nPrec@1 heißt, dass das Target-Attribut im Embedding-Raum nach " \
                          " Kosinus-Ähnlichkeit das ähnlichste Attribut der proijizierten Adjektiv-Nomen-Komposition ist. Prec@5 bedeutet, dass" \
                          " das Attribut noch eines der nächsten 5 Attribute ist.\nTraining auf dem HeiPLAS-dev-set, Test auf dem dem HeiPLAS-test-set."

spearman_explanation = "2. Experiment:\nHier wurde mit einem Datensatz von Mitchell-Lapata gearbeitet, in dem Probanden jeweils die Ähnlichkeit" \
                       " zwischen zwei Adjektiv-Nomen-Phrasen bewertet haben. Ich habe zusätzlich die Ähnlichkeit der Phrasen berechnet, indem" \
                       " ich die Adjektiv-Nomen-Phrasen jeweils in den Embedding-Space abgebildet und danach per Kosinus-Ähnlichkeit verglichen habe." \
                       "\nDanach habe ich die Menschlichen Bewertungen je mit meinen Berechnung per Spearman's r-Korrelation verglichen.\n" \
                       " Außerdem habe ich untersucht, ob die Attribute die \'Source of Similarity\' sein könnten, indem ich mit prec@1 geschaut habe," \
                       " ob die beiden Phrasen das selbe Attribut als ähnlichstes Teilen. In 'average-shared_attribs-top-5' habe ich je geschaut, wie viele" \
                       " Attribute unter den Top-5 beider Phrasen gleich sind (unabhängig von der Platzierung) und dann durch die Anzahl (hier 5) geteilt." \
                       " Das habe ich über alle Attribute, die menschliche Bewertungen über einem gewissen Cut-Off (>4,>5) bekommen haben, gemittelt."


RESULTS_PATH = 'results/'

def perform_complete_test_suite(subset_test_filename, zero_shot_filename, spearman_filename, table_filename, subset = False, zero = False, spearman = False, tables = False, projection_modes = 'best', verbosity = 0):

    print("Subset = {}, Zero = {}, Spearman = {}, Table = {}, projection_modes = {}, Verbosity = {}".format(subset,zero,spearman,tables,projection_modes,verbosity))
    result_dir = RESULTS_PATH + '{}_{}_{}'.format(now.year,now.month,now.day)

    if not '{}_{}_{}'.format(now.year,now.month,now.day) in os.listdir(RESULTS_PATH):
            os.mkdir(result_dir)

    projection_mode_list = PROJECTION_MODE_LIST
    if projection_modes == 'best':
        projection_mode_list = SMALL_PROJECTION_MODE_LSIT
    elif projection_modes == 'medium':
        projection_mode_list = MEDIUM_PROJECTION_LIST
    elif projection_modes == 'test':
        projection_mode_list = TEST_PROJECTION_MODE_LIST

    check_attribute_set_sizes()

    if subset:

        # if not '{}_{}_{}'.format(now.year,now.month,now.day) in os.listdir(RESULTS_PATH):
        #     os.mkdir(result_dir)


        sys.stdout = open(result_dir + '/' + subset_test_filename, 'w')



        sys.stdout.flush()


        print("\n\n----------------------Subset-Tests----------------------".upper())
        perform_subset_tests(train_subsets=ATTRIBUTE_SETS,
                             test_subsets=ATTRIBUTE_SETS,
                             tables=False, quantitive=True,
                             proj_mode_list=projection_mode_list,
                             train_test_exclusivity=False,
                             verbosity=verbosity)

        sys.stdout.flush()

    if zero:
        sys.stdout = open(result_dir + '/' + zero_shot_filename, 'w')

        print("\n\n----------------------Subset-Tests (Testmenge explizit ohne die Attribute aus der Trainingsmenge)----------------------".upper())
        perform_subset_tests(train_subsets=[attribute_set for attribute_set in ATTRIBUTE_SETS if attribute_set[0] != 'ALL'],
                             test_subsets=[ALL_ATTRIBUTES],
                             tables=False, quantitive=True,
                             proj_mode_list=projection_mode_list,
                             train_test_exclusivity=False,
                             verbosity=verbosity)

        sys.stdout.flush()

    if spearman:
        sys.stdout = open(result_dir + '/' + spearman_filename + '.txt', 'w')
        print("\n\n----------------------spearman tests----------------------".upper())
        for attribute_set in ATTRIBUTE_SETS:
            print("\n---------------------Teste mit Attribut-Menge {}----------------".format(attribute_set[0]))
            similarity_experiment(attribute_set,attribute_set,
                                  ['add','nn_weighted_adjective_noun_identity','mult','mitchell_lapata_reversed_2','mitchell_lapata_2'], rating_cutoff_list=[6],
                                  verbosity=verbosity)

        sys.stdout.flush()

        sys.stdout = open(result_dir + '/' + spearman_filename + '_no_tables.txt', 'w')
        print("\n\n----------------------spearman tests----------------------".upper())
        for attribute_set in ATTRIBUTE_SETS:
            print("\n---------------------Teste mit Attribut-Menge {}----------------".format(attribute_set[0]))
            similarity_experiment(attribute_set,attribute_set,
                                  ['add','nn_weighted_adjective_noun_identity','mult','mitchell_lapata_reversed_2','mitchell_lapata_2'], rating_cutoff_list=[0,1,2,3,4,5,6],
                                  tables=False,
                                  verbosity=verbosity)

        sys.stdout.flush()


    if tables:
        sys.stdout = open(result_dir + '/' + table_filename, 'w')

        print("\n\n----------------------Tabellen----------------------".upper())
        perform_subset_tests(train_subsets=ATTRIBUTE_SETS,
                             test_subsets=ATTRIBUTE_SETS,
                             tables=True, quantitive=False,
                             proj_mode_list=projection_mode_list,
                             train_test_exclusivity=False,
                             verbosity=verbosity)

arglist = sys.argv[1:]

# print(arglist)

subset, zero, spearman, tables, verbosity, projection_modes = False, False, False, False, 0, 'all'

if arglist[0] == 'True':
    # print(arglist[0])
    subset = True
if arglist[1] == 'True':
    # print(arglist[1])
    zero = True
if arglist[2] == 'True':
    # print(arglist[2])
    spearman = True
if arglist[3] == 'True':
    # print(arglist[3])
    tables = True

if arglist[4]:
    verbosity = int(arglist[4])

if arglist[5]:
    projection_modes = arglist[5]

perform_complete_test_suite('subset_results.txt','zero_shot_results.txt','spearman_results','tables.txt', subset=subset, zero=zero, spearman=spearman, tables=tables, projection_modes=projection_modes, verbosity = verbosity)


