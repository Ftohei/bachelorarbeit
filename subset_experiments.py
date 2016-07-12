# -*- coding: utf-8 -*-

import sys
from evaluation import evaluate

#NEXT: -Durchläufe mit allen Projektionsmodi machen und schauen wie es aussieht
#Tabellen für alles Berechnen lassen?
#Funktion schreiben, die die Scores mit Matplotlib plottet
#bei den Similarity Tests jeweils die besten Attribute ausgeben lassen

#TODO: Alle Attribute-Sets nochmal aus Matthias Diss holen

CORE_ATTRIBUTES = ('CORE', ['AGE', 'COLOR', 'DIRECTION', 'DURATION', 'SIZE', 'SMELL', 'SPEED', 'TASTE', 'TEMPERATURE', 'WEIGHT'])
MEASUREABLE_ATTRIBUTES = ('MEASUREABLE', ['ABSORBENCY', 'AGE', 'AIRWORTHINESS', 'AUDIBILITY', 'CLARITY', 'CLEANNESS', 'CLEARNESS', 'COLOUR', 'COMPLEXION', 'COMPLEXITY', 'CONSISTENCY', 'CONSTANCY', 'DEHISCENCE', 'DEPTH', 'DIFFERENCE', 'DIRECTION', 'DISTANCE', 'DURATION', 'EFFECTIVENESS', 'EFFICACY', 'EQUALITY', 'FERTILITY', 'FRESHNESS', 'HARDNESS', 'HEALTH', 'HEIGHT', 'INTELLIGENCE', 'LENGTH', 'LIGHT', 'LIKELIHOOD', 'LOGICALITY', 'LUMINOSITY', 'MAGNETISM', 'MATURITY', 'MOTION', 'NUMEROUSNESS', 'OPACITY', 'PITCH', 'POSITION', 'PRICE', 'PURITY', 'QUALITY', 'QUANTITY', 'REPULSION', 'SEAWORTHINESS', 'SENTIENCE', 'SEX', 'SHAPE', 'SHARPNESS', 'SIGNIFICANCE', 'SIMILARITY', 'SIZE', 'SMELL', 'SOCIABILITY', 'SOLIDITY', 'SPEED', 'STRENGTH', 'TASTE', 'TEMPERATURE', 'TEXTURE', 'THICKNESS', 'TYPICALITY', 'VALENCE', 'VOLUME', 'WEIGHT', 'WETNESS', 'WIDTH'])
PROPERTY_ATTRIBUTES = ('PROPERTY', ['ABSORBENCY', 'ABSTEMIOUSNESS', 'ACQUISITIVENESS', 'AGE', 'ANCESTRY', 'ANIMATENESS', 'ANIMATION', 'APPETIZINGNESS', 'ATTENTION', 'AUDIBILITY', 'BOLDNESS', 'BREAKABLENESS', 'COLOR', 'COMPLEXION', 'CONSISTENCY', 'CONTINUITY', 'CUBICITY', 'CURLINESS', 'CURRENTNESS', 'DEGREE', 'DEPTH', 'DESTRUCTIBILITY', 'DISPOSITION', 'DISTANCE', 'DULLNESS', 'DURATION', 'FAIRNESS', 'FRESHNESS', 'FULLNESS', 'HARDNESS', 'HEIGHT', 'IMMEDIACY', 'LENGTH', 'LIGHT', 'LUMINOSITY', 'MAGNITUDE', 'MAJORITY', 'MINORITY', 'MODERATION', 'MODERNITY', 'MUSICALITY', 'NUMEROUSNESS', 'OBVIOUSNESS', 'PERMANENCE', 'PITCH', 'POSITION', 'POWER', 'QUALITY', 'QUANTITY', 'REASONABLENESS', 'REGULARITY', 'SENIORITY', 'SENSITIVITY', 'SENTIENCE', 'SERIOUSNESS', 'SEX', 'SHAPE', 'SHARPNESS', 'SIZE', 'SMELL', 'SOLIDITY', 'SPEED', 'STALENESS', 'STATURE', 'STRENGTH', 'TEMPERATURE', 'TEXTURE', 'THICKNESS', 'TIMING', 'VOLUME', 'WEIGHT', 'WIDTH', 'WILDNESS'])
WEBCHILD_ATTRIBUTES = ('WEBCHILD', ['ABILITY', 'APPEARANCE', 'BEAUTY', 'COLOR', 'EMOTION', 'FEELING', 'LENGTH', 'MOTION', 'QUALITY', 'SENSITIVITY', 'SHAPE', 'SIZE', 'SMELL', 'SOUND', 'STATE', 'STRENGTH', 'TASTE', 'TEMPERATURE', 'WEIGHT'])
SELECTED_ATTRIBUTES = ('SELECTED', ['QUANTITY','WIDTH','CONSISTENCY','AGE','POSITION','LIGHT','COLOR','COMPLEXION','TEMPERATURE','SIZE','SPEED','TEXTURE','WEIGHT','DISTANCE','DEPTH','DURATION','COMPLEXITY','CRISIS','REALITY','IMPORTANCE','NORMALITY','ABSORBENCY','REGULARITY'])

ALL_ATTRIBUTES = ('ALL', ['ABILITY', 'ABSORBENCY', 'ABSTEMIOUSNESS', 'ABSTRACTNESS', 'ACCURACY', 'ACQUISITIVENESS', 'ACTION', 'ACTIVENESS', 'ACTUALITY', 'ADEQUACY', 'ADMISSIBILITY', 'AFFECTEDNESS', 'AGE', 'AIRWORTHINESS', 'ALARM', 'AMBITION', 'ANCESTRY', 'ANIMATENESS', 'ANIMATION', 'APPETIZINGNESS', 'APPROPRIATENESS', 'ASSURANCE', 'ASTRINGENCY', 'ATTENTION', 'ATTENTIVENESS', 'ATTRACTIVENESS', 'ATTRIBUTION', 'AUDIBILITY', 'AUSPICIOUSNESS', 'AWARENESS', 'BEAUTY', 'BEING', 'BENEFICENCE', 'BENIGNITY', 'BOLDNESS', 'BREAKABLENESS', 'CAPABILITY', 'CAREFULNESS', 'CERTAINTY', 'CHANGEABLENESS', 'CHEERFULNESS', 'CIVILITY', 'CLARITY', 'CLEANNESS', 'CLEARNESS', 'COLOUR', 'COMFORT', 'COMMERCE', 'COMMONALITY', 'COMMONNESS', 'COMPLETENESS', 'COMPLEXION', 'COMPLEXITY', 'COMPREHENSIVENESS', 'CONCRETENESS', 'CONFIDENCE', 'CONNECTION', 'CONSISTENCY', 'CONSPICUOUSNESS', 'CONSTANCY', 'CONTINUITY', 'CONVENIENCE', 'CONVENTIONALITY', 'CONVERTIBILITY', 'CORRECTNESS', 'CORRUPTNESS', 'COURAGE', 'COURTESY', 'COWARDICE', 'CREATIVITY', 'CREDIBILITY', 'CRISIS', 'CRITICALITY', 'CUBICITY', 'CURLINESS', 'CURRENTNESS', 'CYCLICITY', 'DEGREE', 'DEHISCENCE', 'DEPTH', 'DESTRUCTIBILITY', 'DIFFERENCE', 'DIFFICULTY', 'DIRECTION', 'DIRECTNESS', 'DISPENSABILITY', 'DISPOSITION', 'DISTANCE', 'DIVERSENESS', 'DOMESTICITY', 'DORMANCY', 'DRAMA', 'DULLNESS', 'DURATION', 'EASE', 'EFFECTIVENESS', 'EFFICACY', 'EMOTIONALITY', 'EQUALITY', 'ESSENTIALITY', 'EVENNESS', 'EVIL', 'EXCITEMENT', 'EXPLICITNESS', 'EXTINCTION', 'FAIRNESS', 'FAMILIARITY', 'FATHERLINESS', 'FEAR', 'FELICITY', 'FERTILITY', 'FIDELITY', 'FINALITY', 'FOREIGNNESS', 'FORMALITY', 'FREEDOM', 'FRESHNESS', 'FRIENDLINESS', 'FULLNESS', 'FUNCTION', 'GENERALITY', 'GENEROSITY', 'GLUTTONY', 'GOOD', 'GREGARIOUSNESS', 'HANDINESS', 'HAPPINESS', 'HARDNESS', 'HEALTH', 'HEIGHT', 'HOLINESS', 'HONESTY', 'HONORABLENESS', 'HUMANENESS', 'HUMANNESS', 'HUMILITY', 'IMMEDIACY', 'IMPORTANCE', 'INDEPENDENCE', 'INDIVIDUALITY', 'INTEGRITY', 'INTELLIGENCE', 'INTENTIONALITY', 'INTEREST', 'INTROSPECTIVENESS', 'INTROVERSION', 'INTRUSIVENESS', 'INWARDNESS', 'KINDNESS', 'LAWFULNESS', 'LEGALITY', 'LENGTH', 'LIGHT', 'LIKELIHOOD', 'LIKENESS', 'LITERACY', 'LIVELINESS', 'LOGICALITY', 'LOYALTY', 'LUMINOSITY', 'MAGNETISM', 'MAGNITUDE', 'MAJORITY', 'MALEFICENCE', 'MALIGNITY', 'MANDATE', 'MATERIALITY', 'MATURITY', 'MEASURE', 'MIND', 'MINDFULNESS', 'MINORITY', 'MODERATION', 'MODERNITY', 'MODESTY', 'MORALITY', 'MOTHERLINESS', 'MOTION', 'MUSICALITY', 'NASTINESS', 'NATURALNESS', 'NATURE', 'NECESSITY', 'NICENESS', 'NOBILITY', 'NORMALITY', 'NUMERACY', 'NUMEROUSNESS', 'OBEDIENCE', 'OBVIOUSNESS', 'OFFENSIVENESS', 'OPACITY', 'ORDINARINESS', 'ORIGINALITY', 'ORTHODOXY', 'OTHERNESS', 'OUTWARDNESS', 'PASSIVITY', 'PERFECTION', 'PERMANENCE', 'PERMISSIVENESS', 'PIETY', 'PITCH', 'PLAYFULNESS', 'PLEASANTNESS', 'POLITENESS', 'POPULARITY', 'POSITION', 'POSSIBILITY', 'POTENCY', 'POTENTIAL', 'POWER', 'PRACTICALITY', 'PRESENCE', 'PRICE', 'PRIDE', 'PROLIXITY', 'PROPRIETY', 'PURITY', 'QUALITY', 'QUANTITY', 'READINESS', 'REALITY', 'REASONABLENESS', 'REASSURANCE', 'RECOGNITION', 'REGULARITY', 'REPULSION', 'REPUTE', 'RESPONSIBILITY', 'RIGHTNESS', 'SAMENESS', 'SARCASM', 'SEAWORTHINESS', 'SENIORITY', 'SENSATIONALISM', 'SENSITIVITY', 'SENTIENCE', 'SEPARATION', 'SERIOUSNESS', 'SEX', 'SHAPE', 'SHARPNESS', 'SIGNIFICANCE', 'SIMILARITY', 'SINCERITY', 'SIZE', 'SMELL', 'SOCIABILITY', 'SOCIALITY', 'SOLIDITY', 'SPEED', 'STALENESS', 'STATURE', 'STATUS', 'STRENGTH', 'SUBSTANTIALITY', 'SUCCESS', 'SUFFICIENCY', 'SUSCEPTIBILITY', 'TAMENESS', 'TASTE', 'TEMPERATURE', 'TEXTURE', 'THICKNESS', 'THOUGHTFULNESS', 'TIMIDITY', 'TIMING', 'TRACTABILITY', 'TRUTH', 'TYPICALITY', 'ULTIMACY', 'UNFAMILIARITY', 'USUALNESS', 'UTILITY', 'VALENCE', 'VIRGINITY', 'VIRTUE', 'VOLUME', 'WARINESS', 'WEIGHT', 'WETNESS', 'WIDTH', 'WILDNESS', 'WORTHINESS'])

PROJECTION_MODE_LIST = ['adj','noun','add','mult','sub_a-n','sub_n-a','avg','max',
                        'mitchell_lapata_0.5', 'mitchell_lapata_reversed_0.5',
                        'mitchell_lapata_1', 'mitchell_lapata_reversed_1',
                        'mitchell_lapata_2', 'mitchell_lapata_reversed_2',
                        'nn_tensor_product_random','nn_tensor_product_identity',
                        'nn_weighted_adjective_identity','nn_weighted_noun_identity',
                        'nn_weighted_adjective_noun_identity','nn_weighted_adjective_random',
                        'nn_weighted_noun_random','nn_weighted_adjective_noun_random',
                        'nn_weighted_adjective_ones', 'nn_weighted_noun_ones','nn_weighted_adjective_noun_ones',
                        'nn_weighted_adj_and_noun_add_identity_with_rands',
                        'nn_weighted_adj_noun_add_sum1_identity','nn_weighted_adj_noun_add_sum1_random',
                        'nn_same_weights_add_identity',

                        ]

MEDIUM_PROJECTION_LIST = ['adj','noun','add','mult','sub_a-n','sub_n-a','avg','max',
                        'mitchell_lapata', 'mitchell_lapata_reversed',
                        'nn_tensor_product_random','nn_tensor_product_identity',
                        'nn_weighted_adjective_identity','nn_weighted_noun_identity',
                        'nn_weighted_adjective_noun_identity',
                        'nn_weighted_adj_noun_add_sum1_identity','nn_weighted_adj_noun_add_sum1_random',
                        'nn_same_weights_add_identity',
                        ]

SMALL_PROJECTION_MODE_LSIT = ['add','mult','sub_a-n','sub_n-a','avg','max','mitchell_lapata',
                        'nn_tensor_product_random',
                        'nn_weighted_adjective_noun_identity',
                        'nn_weighted_adj_and_noun_add_identity_with_rands']

TEST_PROJECTION_MODE_LIST = ['nn_weighted_adjective_noun_identity',]


REALLY_SMALL_PROJ_LIST = ['add','avg','max','nn_weighted_adjective_noun_identity','nn_weighted_adj_and_noun_add_identity_with_rands']

ATTRIBUTE_SETS = [ALL_ATTRIBUTES,CORE_ATTRIBUTES,SELECTED_ATTRIBUTES,MEASUREABLE_ATTRIBUTES,PROPERTY_ATTRIBUTES,WEBCHILD_ATTRIBUTES]

def perform_subset_tests(train_subsets, test_subsets, tables=False, quantitive=True, plot=False, proj_mode_list=PROJECTION_MODE_LIST, train_test_exclusivity = False, verbosity = 2):

    if train_test_exclusivity:
        for train_set in train_subsets:
            for test_set in test_subsets:
                if not train_set[0] == 'ALL' and not (train_set[0] == test_set[0]):     #nicht auf ALL trainieren, da das alle testsets subsummiert UND train und test nicht gleich, da dann auch testmenge = []
                    print("-----------------------")
                    output = "Train: {}, Test: {} \\ {}".format(train_set[0], test_set[0], train_set[0])
                    print(output)
                    evaluate(train_set[1],test_set[1], proj_mode_list, tables=tables, plot=plot, quantitive_eval=quantitive,train_test_exclusivity = train_test_exclusivity, verbosity=verbosity)
                    sys.stdout.flush()

    else:
        for train_set in train_subsets:
            for test_set in test_subsets:
                print("-----------------------")
                output = "Train: %s, Test: %s" % (train_set[0], test_set[0])
                print(output)
                evaluate(train_set[1],test_set[1], proj_mode_list, tables=tables, plot=plot, quantitive_eval=quantitive,train_test_exclusivity = train_test_exclusivity, verbosity=verbosity)
                sys.stdout.flush()

def perform_test(train_set,test_set, tables=False, quantitive=True, plot=False, proj_mode_list=PROJECTION_MODE_LIST, train_test_exclusivity = False, verbosity = 2):
    print("-----------------------")
    output = "Train: %s, Test: %s" % (train_set[0], test_set[0])
    print(output)
    evaluate(train_set[1],test_set[1], proj_mode_list, tables=tables, plot=plot, quantitive_eval=quantitive, train_test_exclusivity = train_test_exclusivity, verbosity=verbosity)
    sys.stdout.flush()

def check_attribute_set_sizes(attribute_list = ATTRIBUTE_SETS):
    for attribute_set in attribute_list:
        print("{}: {}".format(attribute_set[0],len(attribute_set[1])))

# check_attribute_set_sizes()

# perform_subset_tests(train_subsets=ATTRIBUTE_SETS,
#                      test_subsets=ATTRIBUTE_SETS,
#                      tables=False, quantitive=True,
#                      proj_mode_list=SMALL_PROJECTION_MODE_LSIT,
#                      train_test_exclusivity=False,
#                      verbosity=0)



# perform_subset_tests(subsets=[ALL_ATTRIBUTES,CORE_ATTRIBUTES,MEASUREABLE_ATTRIBUTES,PROPERTY_ATTRIBUTES,WEBCHILD_ATTRIBUTES],
#                      tables=False, quantitive=True, verbosity=0)

# perform_test(ALL_ATTRIBUTES,ALL_ATTRIBUTES, tables=False, quantitive=False, plot=True, proj_mode_list=REALLY_SMALL_PROJ_LIST, verbosity=0)

# perform_test(ALL_ATTRIBUTES,ALL_ATTRIBUTES, tables=False, quantitive=True, proj_mode_list=SMALL_PROJECTION_MODE_LSIT, verbosity=0)