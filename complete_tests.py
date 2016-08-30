from similarity_ratings_tests import *
from subset_experiments import *
import sys
import os
import datetime
# Used to compute all tests in a single run

now = datetime.datetime.now()

RESULTS_PATH = 'results/'

def perform_complete_test_suite(subset_test_filename, zero_shot_filename, spearman_filename, table_filename, subset = False, zero = False, spearman = False, tables = False, projection_modes = 'best', verbosity = 0):
    """
    Computes all test necessary for the ba thesis
    :param subset_test_filename: Path for saving subset tests.
    :param zero_shot_filename: Path for saving zero shot tests.
    :param spearman_filename: Path for spearman tests.
    :param table_filename: Path for saving explicit tables.
    :param subset: Flag to include or exclude subset tests from the test suite.
    :param zero: Flag to include or exclude zero shot tests from the test suite.
    :param spearman: Flag to include or exclude spearman tests from the test suite.
    :param tables: Flag to include or exclude explicit tables from the test suite.
    :param projection_modes: certain keywords for specific list of projection modes to test.
    :param verbosity: determines if errors are printed or not.
    :return: null
    """
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
                             train_test_exclusivity=True,
                             verbosity=verbosity)

        sys.stdout.flush()

    if spearman:
        sys.stdout = open(result_dir + '/' + spearman_filename + '.txt', 'w')
        print("\n\n----------------------spearman tests----------------------".upper())
        for attribute_set in ATTRIBUTE_SETS:
            print("\n---------------------Teste mit Attribut-Menge {}----------------".format(attribute_set[0]))
            similarity_experiment(attribute_set,attribute_set,
                                  ['add','nn_weighted_adjective_noun_identity','mult','mitchell_lapata_reversed_2','mitchell_lapata_2'], rating_cutoff_list=[4,5,6],
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

# perform_complete_test_suite('subset_results.txt','zero_shot_results.txt','spearman_results','tables.txt', subset=subset, zero=zero, spearman=spearman, tables=tables, projection_modes=projection_modes, verbosity = verbosity)


check_attribute_set_sizes()



