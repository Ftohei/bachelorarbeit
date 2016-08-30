import matplotlib.pyplot as plt
import numpy as np
from nltk import FreqDist, word_tokenize

dashes = ['--', #    : dashed line
          '-', #     : solid line
          '-.', #   : dash-dot line
          ':', #    : dotted line
           '-',
          '.']

##############SUBSETS für TESTS!###############

results_one_string_subset_test = """
-----------------------
Train: ALL, Test: CORE
                                                     Prec@1    Prec@5
nn_weighted_adjective_identity                       0.85      0.98
mitchell_lapata_reversed_2                           0.73      0.97
nn_weighted_adjective_noun_identity                  0.82      0.97
adj                                                  0.73      0.95
add                                                  0.73      0.95
nn_tensor_product_identity                           0.68      0.91
-----------------------
Train: ALL, Test: SELECTED
                                                     Prec@1    Prec@5
nn_weighted_adjective_noun_identity                  0.72      0.94
nn_weighted_adjective_identity                       0.73      0.90
add                                                  0.56      0.85
mitchell_lapata_reversed_2                           0.58      0.81
adj                                                  0.58      0.76
nn_tensor_product_identity                           0.56      0.74
-----------------------
Train: ALL, Test: MEASUREABLE
                                                     Prec@1    Prec@5
nn_weighted_adjective_noun_identity                  0.64      0.83
nn_weighted_adjective_identity                       0.58      0.81
nn_tensor_product_identity                           0.50      0.64
adj                                                  0.42      0.62
mitchell_lapata_reversed_2                           0.44      0.62
add                                                  0.34      0.62
-----------------------
Train: ALL, Test: PROPERTY
                                                     Prec@1    Prec@5
nn_weighted_adjective_noun_identity                  0.64      0.84
nn_weighted_adjective_identity                       0.58      0.81
nn_tensor_product_identity                           0.50      0.64
adj                                                  0.42      0.60
mitchell_lapata_reversed_2                           0.42      0.59
add                                                  0.32      0.58
-----------------------
Train: ALL, Test: WEBCHILD
                                                     Prec@1    Prec@5
nn_weighted_adjective_noun_identity                  0.77      0.99
nn_weighted_adjective_identity                       0.81      0.96
add                                                  0.47      0.81
nn_tensor_product_identity                           0.58      0.77
adj                                                  0.49      0.74
mitchell_lapata_reversed_2                           0.52      0.74


"""

results_one_string_subset_train = """
----------------------------------------------TEST AUF ALL: Subset training----------------------------------------------

-----------------------
Train: CORE, Test: ALL
                                                     Prec@1    Prec@5
mitchell_lapata_reversed_2                           0.33      0.51
adj                                                  0.33      0.50
nn_weighted_adjective_noun_identity                  0.27      0.47
nn_weighted_adjective_identity                       0.26      0.47
add                                                  0.24      0.45
nn_tensor_product_identity                           0.14      0.21
-----------------------
Train: SELECTED, Test: ALL
                                                     Prec@1    Prec@5
nn_weighted_adjective_noun_identity                  0.31      0.52
mitchell_lapata_reversed_2                           0.33      0.51
adj                                                  0.33      0.50
nn_weighted_adjective_identity                       0.29      0.47
add                                                  0.24      0.45
nn_tensor_product_identity                           0.18      0.27
-----------------------
Train: MEASUREABLE, Test: ALL
                                                     Prec@1    Prec@5
nn_weighted_adjective_noun_identity                  0.37      0.55
nn_weighted_adjective_identity                       0.32      0.53
mitchell_lapata_reversed_2                           0.33      0.51
adj                                                  0.33      0.50
add                                                  0.24      0.45
nn_tensor_product_identity                           0.25      0.35
-----------------------
Train: PROPERTY, Test: ALL
                                                     Prec@1    Prec@5
nn_weighted_adjective_noun_identity                  0.37      0.60
nn_weighted_adjective_identity                       0.36      0.57
mitchell_lapata_reversed_2                           0.33      0.51
adj                                                  0.33      0.50
add                                                  0.24      0.45
nn_tensor_product_identity                           0.28      0.37
-----------------------
Train: WEBCHILD, Test: ALL
                                                     Prec@1    Prec@5
mitchell_lapata_reversed_2                           0.33      0.51
adj                                                  0.33      0.50
nn_weighted_adjective_identity                       0.27      0.49
nn_weighted_adjective_noun_identity                  0.27      0.48
add                                                  0.24      0.45
nn_tensor_product_identity                           0.15      0.22
"""


results_one_string_core_test = """
-----------------------
Train: CORE, Test: CORE
                                                     Prec@1    Prec@5
nn_weighted_adjective_identity                       0.88      0.98
nn_weighted_adjective_noun_identity                  0.88      0.98
mitchell_lapata_reversed_2                           0.73      0.97
adj                                                  0.73      0.95
add                                                  0.73      0.95
nn_tensor_product_identity                           0.65      0.89

-----------------------
Train: SELECTED, Test: CORE
                                                     Prec@1    Prec@5
nn_weighted_adjective_identity                       0.83      0.98
mitchell_lapata_reversed_2                           0.73      0.97
nn_weighted_adjective_noun_identity                  0.85      0.97
adj                                                  0.73      0.95
add                                                  0.73      0.95
nn_tensor_product_identity                           0.64      0.83

-----------------------
Train: MEASUREABLE, Test: CORE
                                                     Prec@1    Prec@5
nn_weighted_adjective_identity                       0.85      0.98
nn_weighted_adjective_noun_identity                  0.89      0.98
mitchell_lapata_reversed_2                           0.73      0.97
adj                                                  0.73      0.95
add                                                  0.73      0.95
nn_tensor_product_identity                           0.65      0.91

-----------------------
Train: PROPERTY, Test: CORE
                                                     Prec@1    Prec@5
mitchell_lapata_reversed_2                           0.73      0.97
nn_weighted_adjective_noun_identity                  0.79      0.97
adj                                                  0.73      0.95
add                                                  0.73      0.95
nn_weighted_adjective_identity                       0.83      0.95
nn_tensor_product_identity                           0.73      0.86

-----------------------
Train: WEBCHILD, Test: CORE
                                                     Prec@1    Prec@5
mitchell_lapata_reversed_2                           0.73      0.97
nn_weighted_adjective_identity                       0.77      0.97
adj                                                  0.73      0.95
add                                                  0.73      0.95
nn_weighted_adjective_noun_identity                  0.74      0.95
nn_tensor_product_identity                           0.53      0.70

"""

results_one_string_selected_test = """
-----------------------
Train: CORE, Test: SELECTED
                                                     Prec@1    Prec@5
add                                                  0.56      0.85
nn_weighted_adjective_identity                       0.56      0.82
nn_weighted_adjective_noun_identity                  0.56      0.82
mitchell_lapata_reversed_2                           0.58      0.81
adj                                                  0.58      0.76
nn_tensor_product_identity                           0.37      0.54

-----------------------
Train: SELECTED, Test: SELECTED
                                                     Prec@1    Prec@5
nn_weighted_adjective_noun_identity                  0.80      0.97
nn_weighted_adjective_identity                       0.80      0.91
add                                                  0.56      0.85
mitchell_lapata_reversed_2                           0.58      0.81
nn_tensor_product_identity                           0.62      0.80
adj                                                  0.58      0.76

-----------------------
Train: MEASUREABLE, Test: SELECTED
                                                     Prec@1    Prec@5
nn_weighted_adjective_noun_identity                  0.73      0.87
nn_weighted_adjective_identity                       0.69      0.86
add                                                  0.56      0.85
mitchell_lapata_reversed_2                           0.58      0.81
adj                                                  0.58      0.76
nn_tensor_product_identity                           0.53      0.71

-----------------------
Train: PROPERTY, Test: SELECTED
                                                     Prec@1    Prec@5
nn_weighted_adjective_noun_identity                  0.66      0.93
nn_weighted_adjective_identity                       0.69      0.89
add                                                  0.56      0.85
mitchell_lapata_reversed_2                           0.58      0.81
adj                                                  0.58      0.76
nn_tensor_product_identity                           0.57      0.69

-----------------------
Train: WEBCHILD, Test: SELECTED
                                                     Prec@1    Prec@5
add                                                  0.56      0.85
mitchell_lapata_reversed_2                           0.58      0.81
nn_weighted_adjective_identity                       0.53      0.81
nn_weighted_adjective_noun_identity                  0.48      0.80
adj                                                  0.58      0.76
nn_tensor_product_identity                           0.31      0.47

"""


results_one_string_measureable_test = """
-----------------------
Train: CORE, Test: MEASUREABLE
                                                     Prec@1    Prec@5
nn_weighted_adjective_identity                       0.38      0.71
nn_weighted_adjective_noun_identity                  0.38      0.68
adj                                                  0.42      0.62
mitchell_lapata_reversed_2                           0.44      0.62
add                                                  0.34      0.62
nn_tensor_product_identity                           0.23      0.36

-----------------------
Train: SELECTED, Test: MEASUREABLE
                                                     Prec@1    Prec@5
nn_weighted_adjective_noun_identity                  0.50      0.70
nn_weighted_adjective_identity                       0.45      0.68
adj                                                  0.42      0.62
mitchell_lapata_reversed_2                           0.44      0.62
add                                                  0.34      0.62
nn_tensor_product_identity                           0.31      0.45

-----------------------
Train: MEASUREABLE, Test: MEASUREABLE
                                                     Prec@1    Prec@5
nn_weighted_adjective_noun_identity                  0.73      0.89
nn_weighted_adjective_identity                       0.65      0.87
nn_tensor_product_identity                           0.62      0.77
adj                                                  0.42      0.62
mitchell_lapata_reversed_2                           0.44      0.62
add                                                  0.34      0.62

-----------------------
Train: PROPERTY, Test: MEASUREABLE
                                                     Prec@1    Prec@5
nn_weighted_adjective_identity                       0.57      0.81
nn_weighted_adjective_noun_identity                  0.58      0.80
adj                                                  0.42      0.62
mitchell_lapata_reversed_2                           0.44      0.62
add                                                  0.34      0.62
nn_tensor_product_identity                           0.45      0.59

-----------------------
Train: WEBCHILD, Test: MEASUREABLE
                                                     Prec@1    Prec@5
nn_weighted_adjective_identity                       0.44      0.74
nn_weighted_adjective_noun_identity                  0.44      0.73
adj                                                  0.42      0.62
mitchell_lapata_reversed_2                           0.44      0.62
add                                                  0.34      0.62
nn_tensor_product_identity                           0.27      0.38
"""

results_one_string_property_test = """
-----------------------
Train: CORE, Test: PROPERTY
                                                     Prec@1    Prec@5
nn_weighted_adjective_identity                       0.37      0.63
nn_weighted_adjective_noun_identity                  0.36      0.61
adj                                                  0.42      0.60
mitchell_lapata_reversed_2                           0.42      0.59
add                                                  0.32      0.58
nn_tensor_product_identity                           0.22      0.33

-----------------------
Train: SELECTED, Test: PROPERTY
                                                     Prec@1    Prec@5
nn_weighted_adjective_noun_identity                  0.43      0.66
nn_weighted_adjective_identity                       0.43      0.62
adj                                                  0.42      0.60
mitchell_lapata_reversed_2                           0.42      0.59
add                                                  0.32      0.58
nn_tensor_product_identity                           0.28      0.42

-----------------------
Train: MEASUREABLE, Test: PROPERTY
                                                     Prec@1    Prec@5
nn_weighted_adjective_noun_identity                  0.52      0.72
nn_weighted_adjective_identity                       0.50      0.70
adj                                                  0.42      0.60
mitchell_lapata_reversed_2                           0.42      0.59
add                                                  0.32      0.58
nn_tensor_product_identity                           0.38      0.53

-----------------------
Train: PROPERTY, Test: PROPERTY
                                                     Prec@1    Prec@5
nn_weighted_adjective_noun_identity                  0.65      0.88
nn_weighted_adjective_identity                       0.62      0.85
nn_tensor_product_identity                           0.58      0.70
adj                                                  0.42      0.60
mitchell_lapata_reversed_2                           0.42      0.59
add                                                  0.32      0.58
-----------------------
Train: WEBCHILD, Test: PROPERTY
                                                     Prec@1    Prec@5
nn_weighted_adjective_identity                       0.40      0.65
nn_weighted_adjective_noun_identity                  0.41      0.65
adj                                                  0.42      0.60
mitchell_lapata_reversed_2                           0.42      0.59
add                                                  0.32      0.58
nn_tensor_product_identity                           0.25      0.35
"""

results_one_string_webchild_test = """
-----------------------
Train: CORE, Test: WEBCHILD
                                                     Prec@1    Prec@5
nn_weighted_adjective_noun_identity                  0.58      0.90
nn_weighted_adjective_identity                       0.61      0.88
add                                                  0.47      0.81
adj                                                  0.49      0.74
mitchell_lapata_reversed_2                           0.52      0.74
nn_tensor_product_identity                           0.43      0.55

-----------------------
Train: SELECTED, Test: WEBCHILD
                                                     Prec@1    Prec@5
nn_weighted_adjective_identity                       0.61      0.90
nn_weighted_adjective_noun_identity                  0.60      0.87
add                                                  0.47      0.81
adj                                                  0.49      0.74
mitchell_lapata_reversed_2                           0.52      0.74
nn_tensor_product_identity                           0.44      0.62


-----------------------
Train: MEASUREABLE, Test: WEBCHILD
                                                     Prec@1    Prec@5
nn_weighted_adjective_identity                       0.82      0.97
nn_weighted_adjective_noun_identity                  0.82      0.97
add                                                  0.47      0.81
nn_tensor_product_identity                           0.62      0.79
adj                                                  0.49      0.74
mitchell_lapata_reversed_2                           0.52      0.74

-----------------------
Train: PROPERTY, Test: WEBCHILD
                                                     Prec@1    Prec@5
nn_weighted_adjective_identity                       0.83      0.97
nn_weighted_adjective_noun_identity                  0.79      0.97
add                                                  0.47      0.81
nn_tensor_product_identity                           0.68      0.78
adj                                                  0.49      0.74
mitchell_lapata_reversed_2                           0.52      0.74
-----------------------
Train: WEBCHILD, Test: WEBCHILD
                                                     Prec@1    Prec@5
nn_weighted_adjective_identity                       0.83      1.00
nn_weighted_adjective_noun_identity                  0.83      1.00
add                                                  0.47      0.81
nn_tensor_product_identity                           0.68      0.77
adj                                                  0.49      0.74
mitchell_lapata_reversed_2                           0.52      0.74
"""


zero_results = """
-----------------------
Train: CORE, Test: ALL \ CORE
                                                     Prec@1    Prec@5
nn_weighted_adjective_noun_identity                  0.27      0.47
add                                                  0.24      0.44
-----------------------
Train: SELECTED, Test: ALL \ SELECTED
                                                     Prec@1    Prec@5
nn_weighted_adjective_noun_identity                  0.27      0.48
add                                                  0.25      0.44
-----------------------
Train: MEASUREABLE, Test: ALL \ MEASUREABLE
                                                     Prec@1    Prec@5
nn_weighted_adjective_noun_identity                  0.29      0.49
add                                                  0.27      0.48
-----------------------
Train: PROPERTY, Test: ALL \ PROPERTY
                                                     Prec@1    Prec@5
nn_weighted_adjective_noun_identity                  0.31      0.50
add                                                  0.28      0.49
-----------------------
Train: WEBCHILD, Test: ALL \ WEBCHILD
                                                     Prec@1    Prec@5
nn_weighted_adjective_noun_identity                  0.27      0.48
add                                                  0.25      0.46
"""


non_zero_results = """
-----------------------
Train: CORE, Test: ALL
                                                     Prec@1    Prec@5
nn_weighted_adjective_noun_identity                  0.27      0.47
add                                                  0.24      0.45
-----------------------
Train: SELECTED, Test: ALL
                                                     Prec@1    Prec@5
nn_weighted_adjective_noun_identity                  0.31      0.52
add                                                  0.24      0.45
-----------------------
Train: MEASUREABLE, Test: ALL
                                                     Prec@1    Prec@5
nn_weighted_adjective_noun_identity                  0.37      0.55
add                                                  0.24      0.45
-----------------------
Train: PROPERTY, Test: ALL
                                                     Prec@1    Prec@5
nn_weighted_adjective_noun_identity                  0.37      0.60
add                                                  0.24      0.45
-----------------------
Train: WEBCHILD, Test: ALL
                                                     Prec@1    Prec@5
nn_weighted_adjective_noun_identity                  0.27      0.48
add                                                  0.24      0.45
"""


sos_quantitaive_eval_results = """
--------------Rating-Cutoff: Human Ratings > 0---------------
Nach Cut-Off werden 1944 verbleibende Phrasen-Paare angeschaut

-------------Source of Similarity: Quantitative Evaluierung-------------
                                             prec@1    average-shared_attribs-top-5
add                                          0.10      0.14
nn_weighted_adjective_noun_identity          0.09      0.11
mult                                         0.00      0.04
mitchell_lapata_reversed_2                   0.16      0.14
mitchell_lapata_2                            0.07      0.11

--------------Rating-Cutoff: Human Ratings > 1---------------
Nach Cut-Off werden 1208 verbleibende Phrasen-Paare angeschaut

-------------Source of Similarity: Quantitative Evaluierung-------------
                                             prec@1    average-shared_attribs-top-5
add                                          0.12      0.18
nn_weighted_adjective_noun_identity          0.12      0.15
mult                                         0.00      0.04
mitchell_lapata_reversed_2                   0.19      0.16
mitchell_lapata_2                            0.09      0.14

--------------Rating-Cutoff: Human Ratings > 2---------------
Nach Cut-Off werden 851 verbleibende Phrasen-Paare angeschaut

-------------Source of Similarity: Quantitative Evaluierung-------------
                                             prec@1    average-shared_attribs-top-5
add                                          0.14      0.21
nn_weighted_adjective_noun_identity          0.16      0.17
mult                                         0.00      0.04
mitchell_lapata_reversed_2                   0.23      0.17
mitchell_lapata_2                            0.10      0.16

--------------Rating-Cutoff: Human Ratings > 3---------------
Nach Cut-Off werden 596 verbleibende Phrasen-Paare angeschaut

-------------Source of Similarity: Quantitative Evaluierung-------------
                                             prec@1    average-shared_attribs-top-5
add                                          0.15      0.23
nn_weighted_adjective_noun_identity          0.19      0.20
mult                                         0.00      0.04
mitchell_lapata_reversed_2                   0.27      0.18
mitchell_lapata_2                            0.11      0.19

--------------Rating-Cutoff: Human Ratings > 4---------------
Nach Cut-Off werden 397 verbleibende Phrasen-Paare angeschaut

-------------Source of Similarity: Quantitative Evaluierung-------------
                                             prec@1    average-shared_attribs-top-5
add                                          0.18      0.27
nn_weighted_adjective_noun_identity          0.22      0.25
mult                                         0.00      0.04
mitchell_lapata_reversed_2                   0.33      0.20
mitchell_lapata_2                            0.14      0.22

--------------Rating-Cutoff: Human Ratings > 5---------------
Nach Cut-Off werden 200 verbleibende Phrasen-Paare angeschaut

-------------Source of Similarity: Quantitative Evaluierung-------------
                                             prec@1    average-shared_attribs-top-5
add                                          0.20      0.32
nn_weighted_adjective_noun_identity          0.26      0.33
mult                                         0.00      0.04
mitchell_lapata_reversed_2                   0.41      0.21
mitchell_lapata_2                            0.18      0.30

--------------Rating-Cutoff: Human Ratings > 6---------------
Nach Cut-Off werden 82 verbleibende Phrasen-Paare angeschaut

-------------Source of Similarity: Quantitative Evaluierung-------------
                                             prec@1    average-shared_attribs-top-5
add                                          0.23      0.39
nn_weighted_adjective_noun_identity          0.29      0.42
mult                                         0.00      0.06
mitchell_lapata_reversed_2                   0.49      0.25
mitchell_lapata_2                            0.33      0.40
"""


manual_sos_string = """

AST_AMOUNT                      :   LARGE_QUANTITY                		Ja
QUANTITY                  0.58   :   QUANTITY                  0.78
SIZE                      0.57   :   SIZE                      0.63
WIDTH                     0.42   :   WIDTH                     0.41
THICKNESS                 0.38   :   THICKNESS                 0.39
substantiality            0.35   :   volume                    0.36

IMPORTANT_PART                   :   SIGNIFICANT_ROLE              		ja
IMPORTANCE                0.63   :   SIGNIFICANCE              0.55
SIGNIFICANCE              0.54   :   IMPORTANCE                0.53
necessity                 0.41   :   status                    0.41
pride                     0.39   :   RESPONSIBILITY            0.36
RESPONSIBILITY            0.36   :   likelihood                0.35

LARGE_NUMBER                     :   VAST_AMOUNT                  		ja
SIZE                      0.68   :   QUANTITY                  0.58
QUANTITY                  0.46   :   SIZE                      0.57
WIDTH                     0.38   :   WIDTH                     0.42
volume                    0.38   :   thickness                 0.38
complexity                0.34   :   substantiality            0.35

GENERAL_PRINCIPLE                :   BASIC_RULE                   		naja
GENERALITY                0.61   :   GENERALITY                0.44
concreteness              0.36   :   humaneness                0.43
humanness                 0.34   :   practicality              0.43
equality                  0.34   :   reasonableness            0.42
morality                  0.34   :   typicality                0.41

EFFECTIVE_WAY                    :   EFFICIENT_USE                 		ja
EFFECTIVENESS             0.75   :   EFFECTIVENESS             0.47
efficacy                  0.48   :   PRACTICALITY              0.39
potency                   0.43   :   ACCURACY                  0.38
PRACTICALITY              0.42   :   consistency               0.35
ACCURACY                  0.41   :   quality                   0.35

FEDERAL_ASSEMBLY                 :   NATIONAL_GOVERNMENT           		nein, mögliches beispiel zur besprechung
nature                    0.22   :   importance                0.35
seniority                 0.22   :   significance              0.31
majority                  0.22   :   stature                   0.31
otherness                 0.22   :   cleanness                 0.30
ordinariness              0.21   :   seriousness               0.30

CERTAIN_CIRCUMSTANCE             :   PARTICULAR_CASE               		nein
COMMONNESS                0.41   :   significance              0.48
otherness                 0.39   :   importance                0.46
foreignness               0.37   :   COMMONNESS                0.44
concreteness              0.36   :   seriousness               0.40
materiality               0.36   :   abstractness              0.40

VAST_AMOUNT                      :   HIGH_PRICE                    		nein
QUANTITY                  0.58   :   degree                    0.54
size                      0.57   :   height                    0.49
width                     0.42   :   price                     0.45
thickness                 0.38   :   moderation                0.44
substantiality            0.35   :   QUANTITY                  0.32

CERTAIN_CIRCUMSTANCE             :   PARTICULAR_CASE               		nein
COMMONNESS                0.41   :   significance              0.48
otherness                 0.39   :   importance                0.46
foreignness               0.37   :   COMMONNESS                0.44
concreteness              0.36   :   seriousness               0.40
materiality               0.36   :   abstractness              0.40

DIFFERENT_KIND                   :   VARIOUS_FORM                 		naja
sameness                  0.56   :   OTHERNESS                 0.39
similarity                0.55   :   COMMONALITY               0.34
DIFFERENCE                0.51   :   DIFFERENCE                0.33
COMMONALITY               0.50   :   materiality               0.33
OTHERNESS                 0.48   :   foreignness               0.33

SOCIAL_EVENT                     :   SPECIAL_CIRCUMSTANCE          		nein
sociality                 0.46   :   ordinariness              0.45
sociability               0.42   :   commonness                0.43
equality                  0.35   :   OTHERNESS                 0.38
OTHERNESS                 0.31   :   nobility                  0.36
importance                0.31   :   gregariousness            0.35

SPECIAL_CIRCUMSTANCE             :   PARTICULAR_CASE               		nein
ordinariness              0.45   :   significance              0.48
COMMONNESS                0.43   :   importance                0.46
otherness                 0.38   :   COMMONNESS                0.44
nobility                  0.36   :   seriousness               0.40
gregariousness            0.35   :   abstractness              0.40

OLD_PERSON                       :   ELDERLY_LADY                  		naja
AGE                       0.44   :   commonness                0.31
otherness                 0.40   :   health                    0.30
typicality                0.40   :   AGE                       0.30
abstractness              0.38   :   thoughtfulness            0.30
beauty                    0.36   :   attentiveness             0.29

LARGE_QUANTITY                   :   GREAT_MAJORITY                		ja
quantity                  0.78   :   majority                  0.70
SIZE                      0.63   :   minority                  0.42
width                     0.41   :   quality                   0.41
thickness                 0.39   :   SIZE                      0.32
volume                    0.36   :   cleanness                 0.31

MAJOR_ISSUE                      :   AMERICAN_COUNTRY              		nein
significance              0.45   :   foreignness               0.32
importance                0.44   :   corruptness               0.28
size                      0.36   :   niceness                  0.25
stature                   0.35   :   cleanness                 0.25
complexity                0.33   :   humaneness                0.24

CERTAIN_CIRCUMSTANCE             :   PARTICULAR_CASE               		nein
COMMONNESS                0.41   :   significance              0.48
otherness                 0.39   :   importance                0.46
foreignness               0.37   :   COMMONNESS                0.44
concreteness              0.36   :   seriousness               0.40
materiality               0.36   :   abstractness              0.40

NEW_INFORMATION                  :   FURTHER_EVIDENCE              		nein
individuality             0.46   :   credibility               0.37
permanence                0.44   :   possibility               0.35
immediacy                 0.43   :   finality                  0.35
originality               0.43   :   sharpness                 0.35
correctness               0.42   :   depth                     0.33

EARLIER_WORK                     :   EARLY_STAGE                  		ja
TIMING                    0.33   :   TIMING                    0.62
otherness                 0.32   :   height                    0.36
accuracy                  0.32   :   nature                    0.30
regularity                0.32   :   maturity                  0.30
ordinariness              0.31   :   seriousness               0.29

NEW_LIFE                         :   EARLY_AGE                     		nein, mögliches beispiel zur besprechung
permanence                0.45   :   age                       0.62
individuality             0.44   :   timing                    0.58
freshness                 0.43   :   maturity                  0.33
concreteness              0.42   :   height                    0.32
abstractness              0.42   :   tameness                  0.31
"""

# These are the "Tableau 20" colors as RGB.
tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]

FIGSIZE = (10,6)

strong_colours = [(228,26,28), (55,126,184), (77,175,74), (152,78,163), (255,127,0), (0,238,238), (166,86,40), (247,129,191), (153,153,153)]

# Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.
for i in range(len(tableau20)):
    r, g, b = tableau20[i]
    tableau20[i] = (r / 255., g / 255., b / 255.)

for i in range(len(strong_colours)):
    r, g, b = strong_colours[i]
    strong_colours[i] = (r / 255., g / 255., b / 255.)


models = ['nn_weighted_adjective_noun_identity','nn_weighted_adjective_identity','add','nn_tensor_product_identity','adj','mitchell_lapata_reversed_2']
nice_labels = ['Distinctly Weighted Noun and Adjective', 'Weighted Adjective', 'Vector Addition', 'Tensor Product', 'Adjective', 'Dilating the Noun']
subsets = ['CORE','MEASUREABLE','PROPERTY','WEBCHILD']
axarr_titles = ['CORE','SELECTED','MEASUREABLE','PROPERTY','WEBCHILD']
# words = results_one_string_subset_test.split()

# print(words)

def extract_data_from_string(string):
    words = string.split()


    result_dict = {}
    for model in models:
        result_dict[model] = []

    current_train = ''
    current_test = ''

    for i in range(0,len(words)):
        if words[i].lower() in ['train:']:
            current_train = words[i+1]
        if words[i].lower() in ['test:']:
            current_test = words[i+1]
        if words[i] in models:
            result_dict[words[i]].append((current_train[:-1],
                                          current_test,
                                          float(words[i+1]),
                                          float(words[i+2])))

    for model in result_dict:
        # print(model,result_dict[model])
        pass

    return result_dict

def print_p1_p5(path, data_string, evaluate_for = 'test', name = '', baselines = {}):




    legend_size = 8

    plt.clf()
    plt.figure(figsize=FIGSIZE)

    print("--------------------Evaluiere {}".format(evaluate_for).upper())

    result_dict = extract_data_from_string(data_string)

    print(result_dict)

    if evaluate_for == 'test':
        plt.xlabel('Test Sets')
    elif evaluate_for == 'train':
        plt.xlabel('Training Sets')
    else:
        print("Weder Train noch Test spezifiziert!")
        quit(9)

    plt.ylabel('P@1')
    plt.xlim([-0.5,4.5])
    plt.ylim([0,1])

    for i in range(0,5):
        plt.axvline(i, color='0.25',linestyle=':')
    color = '0'

    labels = []
    if evaluate_for == 'test':
        labels = [testset for trainset,testset,p_1,p_5 in result_dict[models[1]]]
    elif evaluate_for == 'train':
        labels = [trainset for trainset,testset,p_1,p_5 in result_dict[models[1]]]
    num_subsets = len(labels)

    line_type = 0
    for model in models:
        test_results_p1 = [p_1 for trainset, testset, p_1, p_5 in result_dict[model]]
        plt.plot(list(range(0,5)), test_results_p1, '-', c=strong_colours[line_type], label=nice_labels[line_type], linewidth=2)
        plt.xticks(list(range(0,5)), labels, rotation=15)
        line_type += 1

    if baselines:
        for model in list(baselines):
            # print(line_type, tableau20[line_type])
            test_results_p1 = [p_1 for trainset, testset, p_1, p_5 in baselines[model]]
            plt.plot(list(range(0,4)), test_results_p1, '-', c=strong_colours[line_type], label=model + " (Baseline)", linewidth=2)
            # plt.xticks(list(range(0,4)), labels, rotation=15)
            line_type += 1

    if name or evaluate_for == 'test':
       plt.legend(loc=3,prop={'size':legend_size})
    elif evaluate_for == 'train':
       plt.legend(loc=1,prop={'size':legend_size})
    plt.gcf().subplots_adjust(bottom=0.2)

    if name:
        plt.savefig(path + "/" + name + "_p1.pdf")
    else:
        plt.savefig(path + "/subset_" + evaluate_for + "_p1.pdf")
    plt.ylabel('@5')

    plt.clf()
    plt.figure(figsize=FIGSIZE)

    if evaluate_for == 'test':
        plt.xlabel('Test Sets')
    elif evaluate_for == 'train':
        plt.xlabel('Training Sets')
    else:
        print("Weder Train noch Test spezifiziert!")
        quit(9)
    plt.ylabel('P@5')
    plt.xlim([-0.5,4.5])
    plt.ylim([0,1])

    for i in range(0,5):
        plt.axvline(i, color='0.25',linestyle=':')

    line_type = 0
    for model in models:
        test_results_p5 = [p_5 for trainset,testset, p_1, p_5 in result_dict[model]]
        plt.plot(list(range(0,5)), test_results_p5, '-', c=strong_colours[line_type], label=nice_labels[line_type], linewidth=2)
        plt.xticks(list(range(0,5)), labels, rotation=15)
        line_type += 1

    # if baselines:
    #     # line_type = 0
    #     for model in list(baselines):
    #         test_results_p5 = [p_5 for trainset, testset, p_1, p_5 in baselines[model]]
    #         plt.plot(list(range(0,4)), test_results_p5, '-', c=tableau20[line_type], label=model + " (Baseline)")
    #         plt.xticks(list(range(0,4)), labels, rotation=15)
    #         line_type += 1

    if name or evaluate_for == 'test':
        plt.legend(loc=3,prop={'size':legend_size})
    elif evaluate_for == 'train':
       plt.legend(loc=1,prop={'size':legend_size})

    plt.gcf().subplots_adjust(bottom=0.2)

    if name:
        plt.savefig(path + "/" + name + "_p5.pdf")
    else:
        plt.savefig(path + "/subset_" + evaluate_for + "_p5.pdf")


def print_ax_arr_test_sets(path, data_strings, name):

    plt.close('all')

    tableau2 = [tableau20[0],tableau20[2]]
    strong2 = [strong_colours[0], strong_colours[2]]
    nice_labls = [nice_labels[0],nice_labels[2]]

    legend_size = 8

    plt.clf()



    f, axarr = plt.subplots(1, 5, figsize=(20  , 6  ))

    # axarr[0,0].set_xlabel('Training Sets')
    axarr[0].set_ylabel('P@1')
    # axarr[1,0].set_ylabel('Precision@1')

    k = 0
    l = 0
    for i in range(0,5):
        # if l >= 3:
        #     k = 1
        #     l = 0
        l = i

        axarr[i].set_title(axarr_titles[i])

        result_dict = extract_data_from_string(data_strings[i])

        labels = [trainset for trainset,testset,p_1,p_5 in result_dict[models[1]]]

        # print(len(labels),labels)

        # print(axarr[k,l].get_xticklabels())
        axarr[i].set_ylim([0,1])
        axarr[i].set_xlim([-0.5,4.5])

        axarr[i].set_xticks(np.arange(0,5,1))
        axarr[i].set_xticklabels(labels)

        for j in range(0,5):
            axarr[i].axvline(j, color='0.25',linestyle=':')
        # axarr[i].axvline()

        line_type = 0
        for model in ['nn_weighted_adjective_noun_identity','add']:
            test_results_p1 = [p_1 for trainset, testset, p_1, p_5 in result_dict[model]]
            axarr[i].plot(list(range(0,5)), test_results_p1, '-', c=strong2[line_type], label=nice_labls[line_type], linewidth=2)
            line_type += 1
        # l+=1

    # axarr[1,2].set_visible(False)

    # plt.legend(loc=3,prop={'size':legend_size})

    # # Fine-tune figure; hide x ticks for top plots and y ticks for right plots
    # plt.setp([a.get_xticklabels() for a in [axarr[0, 0],axarr[0,1]]], visible=False)
    plt.setp([a.get_xticklabels() for a in axarr[:]], rotation=90)
    plt.setp([a.get_yticklabels() for a in axarr[:]], visible=False)
    plt.setp([a.get_yticklabels() for a in [axarr[0]]], visible=True)


    # plt.ylabel("Precision@1")
    # plt.xlabel("Training Sets")
    # f.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    # plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    handles, labels = axarr[i].get_legend_handles_labels()
    # print(handles, labels)
    plt.figlegend(handles,labels,loc='lower center')
    # axarr[0,0].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.gcf().subplots_adjust(bottom=0.4 )
    plt.savefig(path + "/" + name + "_axarr.pdf", bbox_inches='tight')

def print_zero_vs_non_zero(path, zero_string, non_zero_string, name, p5 = False):


    legend_size = 8

    plt.clf()
    plt.figure(figsize=FIGSIZE)

    tableau2 = [tableau20[0],tableau20[2]]
    strong2 = [strong_colours[0], strong_colours[2]]
    nice_labls = [nice_labels[0],nice_labels[2]]

    zero_result_dict = extract_data_from_string(zero_string)
    non_zero_result_dict = extract_data_from_string(non_zero_string)

    plt.xlabel('Training Sets')

    if p5:
        plt.ylabel('P@5')
    else:
        plt.ylabel('P@1')
    plt.xlim([-0.5,4.5])
    plt.ylim([0,1])

    for i in range(0,5):
        plt.axvline(i, color='0.25',linestyle=':')

    labels = [trainset for trainset,testset,p_1,p_5 in zero_result_dict['add']]
    # num_subsets = len(labels)

    plt.xticks(list(range(0,5)), labels, rotation=15)

    print(labels)

    line_type = 0
    for model in ['nn_weighted_adjective_noun_identity','add']:
        # zero_test_results = []
        # non_zero_test_results = []
        if p5:
            zero_test_results = [p_5 for trainset,testset, p_1, p_5 in zero_result_dict[model]]
            non_zero_test_results = [p_5 for trainset,testset, p_1, p_5 in non_zero_result_dict[model]]
        else:
            zero_test_results = [p_1 for trainset,testset, p_1, p_5 in zero_result_dict[model]]
            non_zero_test_results = [p_1 for trainset,testset, p_1, p_5 in non_zero_result_dict[model]]

        # print(p5,zero_test_results)
        # print(p5,non_zero_test_results)
        plt.plot(list(range(0,5)), non_zero_test_results, '-', c=strong2[line_type], label=nice_labls[line_type], linewidth=2)
        plt.plot(list(range(0,5)), zero_test_results, '--', c=strong2[line_type], label=nice_labls[line_type] + " Zero-Shot", linewidth=2)
        # plt.xticks(list(range(0,5)), labels, rotation=15)
        line_type += 1

    # plt.figure(figsize=(10,10))
    plt.legend(loc=3,prop={'size':legend_size})
    plt.gcf().subplots_adjust(bottom=0.2)

    if p5:
        plt.savefig(path + "/" + name + "_p5.pdf")
    else:
        plt.savefig(path + "/" + name + "_p1.pdf")

baseline_models = {'C-LDA with Multiplication':[('ALL','CORE',0.6,0.0),('ALL','SELECTED',0.41,0.0),('ALL','MEASUREABLE',0.25,0.0),('ALL','PROPERTY',0.22,0.0)],
                   'L-LDA with Multiplication':[('ALL','CORE',0.68,0.0),('ALL','SELECTED',0.51,0.0),('ALL','MEASUREABLE',0.27,0.0),('ALL','PROPERTY',0.25,0.0)]}

print_p1_p5("/Users/Fabian/Documents/Uni/6. Semester/Bachelorarbeit/tex/grafiken_ba",
            results_one_string_subset_test, 'test', baselines=baseline_models)




print_p1_p5("/Users/Fabian/Documents/Uni/6. Semester/Bachelorarbeit/tex/grafiken_ba",
            results_one_string_subset_train, 'train')


print_p1_p5("/Users/Fabian/Documents/Uni/6. Semester/Bachelorarbeit/tex/grafiken_ba",
            results_one_string_core_test, 'train', name='test_on_core')


print_p1_p5("/Users/Fabian/Documents/Uni/6. Semester/Bachelorarbeit/tex/grafiken_ba",
            results_one_string_selected_test, 'train', name='test_on_selected')

print_p1_p5("/Users/Fabian/Documents/Uni/6. Semester/Bachelorarbeit/tex/grafiken_ba",
            results_one_string_measureable_test, 'train', name='test_on_measureable')

print_p1_p5("/Users/Fabian/Documents/Uni/6. Semester/Bachelorarbeit/tex/grafiken_ba",
            results_one_string_property_test, 'train', name='test_on_property')

print_p1_p5("/Users/Fabian/Documents/Uni/6. Semester/Bachelorarbeit/tex/grafiken_ba",
            results_one_string_webchild_test, 'train', name='test_on_webchild')

print_ax_arr_test_sets("/Users/Fabian/Documents/Uni/6. Semester/Bachelorarbeit/tex/grafiken_ba",
                       [results_one_string_core_test,results_one_string_selected_test,
                        results_one_string_measureable_test,results_one_string_property_test,
                        results_one_string_webchild_test], 'all_train_sets')

print_zero_vs_non_zero("/Users/Fabian/Documents/Uni/6. Semester/Bachelorarbeit/tex/grafiken_ba",
                       zero_string=zero_results, non_zero_string=non_zero_results, name="zero_vs_nonzero")
print_zero_vs_non_zero("/Users/Fabian/Documents/Uni/6. Semester/Bachelorarbeit/tex/grafiken_ba",
                       zero_string=zero_results, non_zero_string=non_zero_results, p5=True, name="zero_vs_nonzero")


correlation_trainingset_results=[
    ('ALL ATTRIBUTES',0.443),
    ('CORE',0.499),
    ('SELECTED',0.505),
    ('MEASUREABLE', 0.496),
    ('PROPERTY',0.492),
    ('WEBCHILD',0.482),
]

def correlation_training_set_plot(path,corr_results,name):
    legend_size = 9

    plt.clf()




    plt.xlabel('Training Sets')
    plt.ylabel('Spearman\'s $\\rho$')
    plt.xlim([-0.5,5.5])
    plt.ylim([0,1])

    for i in range(0,6):
        plt.axvline(i, color='0.25',linestyle=':')

    labels = [trainset for trainset,rho in corr_results]
    values = [rho for trainset,rho in corr_results]
    plt.xticks(list(range(0,6)), labels, rotation=15)

    plt.plot(list(range(0,6)), values, '-', c=strong_colours[0], label='Distinctly Weighted Noun and Adjective', linewidth=2)
    plt.plot(list(range(0,6)), [0.48,0.48,0.48,0.48,0.48,0.48], '-', c=strong_colours[2], label='Vector Addition', linewidth=2)

    plt.legend(loc=3,prop={'size':legend_size})
    plt.gcf().subplots_adjust(bottom=0.2)

    plt.savefig(path + "/" + name + ".pdf")

correlation_training_set_plot("/Users/Fabian/Documents/Uni/6. Semester/Bachelorarbeit/tex/grafiken_ba", correlation_trainingset_results, "correlation_trainingset_increases")




models = ['nn_weighted_adjective_noun_identity', 'add','mult','mitchell_lapata_2','mitchell_lapata_reversed_2']
nice_labels = ['Distinctly Weighted Noun and Adjective', 'Vector Addition', 'Vector Multiplication', 'Dilating the Adjective', 'Dilating the Noun']


def parse_sos_strings(sos_string):
    words = sos_string.split()
    # print(words)
    print("----------parse SOS string-------")
    current_cutoff = '>0'

    d = {}
    for model in models:
        d[model] = []

    for i in range(0,len(words)):
        if words[i] == '>':
            current_cutoff = words[i] + words[i+1][0]
        if words[i] in models:
            d[words[i]].append((current_cutoff,words[i+1],words[i+2]))

    for key in list(d):
        print(key, d[key])

    return d
# parse_sos_strings(sos_quantitaive_eval_results)

# models = ['nn_weighted_adjective_noun_identity','nn_weighted_adjective_identity','add','nn_tensor_product_identity','adj','mitchell_lapata_reversed_2']
# nice_labels = ['Distinctly weighted Noun and Adjective', 'Weighted Adjective', 'Vector Addition', 'Tensor Product', 'Adjective', 'Dilating the Noun']


def print_quantitative_eval_results(path, string, name, p5=False):
    data = parse_sos_strings(string)

    tableau5 = [tableau20[0],tableau20[2], tableau20[10], tableau20[5], tableau20[18 ]]
    strong5 = [strong_colours[0],strong_colours[2], strong_colours[3], strong_colours[4], strong_colours[5 ]]


    # for model in models:
    #     print(model, data[model])

    legend_size = 9



    plt.clf()

    plt.xlabel('Rating Cutoff')

    if p5:
        plt.ylabel('Average Shared Top 5 Attributes')
    else:
        plt.ylabel('Average Shared Top Attribute')
    plt.xlim([-0.5,6.5])
    plt.ylim([0,1])

    x_range = range(0,7)

    for i in x_range:
        plt.axvline(i, color='0.25',linestyle=':')

    labels = [cutoff for cutoff,p_1,p_5 in data['add']]
    # num_subsets = len(labels)

    plt.xticks(x_range, labels, rotation=0)

    # print(labels)

    line_type = 0
    for model in models:
        # zero_test_results = []
        # non_zero_test_results = []
        results = []
        if p5:
            results = [p_5 for cutoff,p_1,p_5 in data[model]]
        else:
            results = [p_1 for cutoff,p_1,p_5 in data[model]]
        # print(len(results), results)

        print(len(x_range), len(results))
        plt.plot(x_range, results, '-', c=strong5[line_type], label=nice_labels[line_type], linewidth=2)
        line_type += 1

    plt.legend(loc='upper left',prop={'size':legend_size})
    plt.gcf().subplots_adjust(bottom=0.2)

    if p5:
        plt.savefig(path + "/" + name + "_p5.pdf")
    else:
        plt.savefig(path + "/" + name + "_p1.pdf")

def print_quantitative_eval_results_one_plot(path, string, name):
    data = parse_sos_strings(string)
    print("-----------in plot methode--------")

    print(data)

    labels = [cutoff for cutoff,p_1,p_5 in data['add']]
    strong5 = [strong_colours[0],strong_colours[2], strong_colours[3], strong_colours[4], strong_colours[5]]

    # for model in models:
    #     print(model, data[model])

    legend_size = 9

    x_range = range(0,7)

    plt.clf()

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))

    # ax1.setTitle('Average Shared Top Attribute')
    # ax2.setTitle('Average Shared Top 5 Attributes')
    ax1.set_ylabel('Average Shared Top Attribute')
    ax2.set_ylabel('Average Shared Top 5 Attributes')

    ax1.set_xlabel('Rating Cutoff')
    ax2.set_xlabel('Rating Cutoff')

    for ax in ax1,ax2:
        ax.set_xlim([-0.5,6.5])
        ax.set_xticks(np.arange(0,7,1))
        ax.set_xticklabels(labels)
        ax.set_ylim([0,1])

        for i in x_range:
            ax.axvline(i, color='0.25',linestyle=':')


    # num_subsets = len(labels)

        # ax.xticks(x_range, labels, rotation=0)

        # print(labels)

    print(models)

    line_type = 0

    for model in models:
        print(model, data[model])


    for model in models:
        # print(model)
        # print(data[model])
        results_top1 = [p_1 for cutoff,p_1,p_5 in data[model]]
        results_top5 = [p_5 for cutoff,p_1,p_5 in data[model]]
        # print(len(results), results)

        print(len(x_range), len(results_top1), len(results_top5))

        ax2.plot(x_range, results_top5, '-', c=strong5[line_type], label=nice_labels[line_type],linewidth=2)
        ax1.plot(x_range, results_top1, '-', c=strong5[line_type], label=nice_labels[line_type],linewidth=2)
        line_type += 1

    handles, labels = ax1.get_legend_handles_labels()
    print(labels)

    # plt.figlegend(handles,labels,loc='upper right ', prop={'size':legend_size})
    ax1.legend(handles, labels, loc='upper left',prop={'size':9})
    plt.gcf().subplots_adjust(bottom=0.2)
    plt.tight_layout()

    plt.savefig(path + "/" + name + "_oneplot.pdf")


GRAPHICS_PATH = "/Users/Fabian/Documents/Uni/6. Semester/Bachelorarbeit/tex/grafiken_ba"

# print_quantitative_eval_results(GRAPHICS_PATH, sos_quantitaive_eval_results, "sos_quantitative")
# print_quantitative_eval_results(GRAPHICS_PATH, sos_quantitaive_eval_results, "sos_quantitative", p5=True)
print_quantitative_eval_results_one_plot(GRAPHICS_PATH, sos_quantitaive_eval_results, "sos_quantitative")


# freqs = FreqDist(word_tokenize(manual_sos_string))
#
# print("Ja ",freqs['ja'])
# print("Nein ",freqs['nein'])
# print("Naja ", freqs['naja'])



# complete_human_ratings_string = """
# VAST_AMOUNT                      :   LARGE_QUANTITY                (nn_weighted_adjective_noun_identity)
# SIZE                      0.67   :   QUANTITY                  0.83
# QUANTITY                  0.64   :   SIZE                      0.70
# WIDTH                     0.45   :   WIDTH                     0.45
# COMPLEXITY                0.35   :   COMPLEXITY                0.33
# DEPTH                     0.34   :   DEPTH                     0.29
#
# VAST_AMOUNT                      :   LARGE_QUANTITY                (nn_weighted_adjective_noun_identity)
# SIZE                      0.67   :   QUANTITY                  0.83
# QUANTITY                  0.64   :   SIZE                      0.70
# WIDTH                     0.45   :   WIDTH                     0.45
# COMPLEXITY                0.35   :   COMPLEXITY                0.33
# DEPTH                     0.34   :   DEPTH                     0.29
#
# IMPORTANT_PART                   :   SIGNIFICANT_ROLE              (nn_weighted_adjective_noun_identity)
# IMPORTANCE                0.64   :   IMPORTANCE                0.48
# COMPLEXITY                0.29   :   position                  0.41
# duration                  0.20   :   size                      0.30
# normality                 0.19   :   COMPLEXITY                0.26
# crisis                    0.19   :   depth                     0.20
#
# LARGE_NUMBER                     :   VAST_AMOUNT                   (nn_weighted_adjective_noun_identity)
# SIZE                      0.75   :   SIZE                      0.67
# QUANTITY                  0.47   :   QUANTITY                  0.64
# WIDTH                     0.42   :   WIDTH                     0.45
# COMPLEXITY                0.33   :   COMPLEXITY                0.35
# weight                    0.31   :   depth                     0.34
#
# GENERAL_PRINCIPLE                :   BASIC_RULE                    (nn_weighted_adjective_noun_identity)
# IMPORTANCE                0.36   :   IMPORTANCE                0.38
# NORMALITY                 0.25   :   NORMALITY                 0.35
# position                  0.23   :   COMPLEXITY                0.27
# REALITY                   0.21   :   REALITY                   0.26
# COMPLEXITY                0.16   :   consistency               0.25
#
# EFFECTIVE_WAY                    :   EFFICIENT_USE                 (nn_weighted_adjective_noun_identity)
# CONSISTENCY               0.27   :   speed                     0.38
# position                  0.25   :   CONSISTENCY               0.29
# COMPLEXITY                0.25   :   COMPLEXITY                0.28
# importance                0.25   :   REGULARITY                0.21
# REGULARITY                0.23   :   absorbency                0.20
#
# FEDERAL_ASSEMBLY                 :   NATIONAL_GOVERNMENT           (nn_weighted_adjective_noun_identity)
# SIZE                      0.16   :   importance                0.27
# speed                     0.15   :   SIZE                      0.23
# position                  0.14   :   crisis                    0.22
# DISTANCE                  0.13   :   quantity                  0.19
# weight                    0.13   :   DISTANCE                  0.18
#
# CERTAIN_CIRCUMSTANCE             :   PARTICULAR_CASE               (nn_weighted_adjective_noun_identity)
# REGULARITY                0.30   :   importance                0.41
# normality                 0.25   :   size                      0.32
# position                  0.25   :   complexity                0.31
# reality                   0.24   :   duration                  0.30
# crisis                    0.20   :   REGULARITY                0.26
#
# VAST_AMOUNT                      :   HIGH_PRICE                    (nn_weighted_adjective_noun_identity)
# SIZE                      0.67   :   QUANTITY                  0.45
# QUANTITY                  0.64   :   SIZE                      0.34
# WIDTH                     0.45   :   temperature               0.32
# complexity                0.35   :   speed                     0.30
# depth                     0.34   :   WIDTH                     0.29
#
# IMPORTANT_PART                   :   SIGNIFICANT_ROLE              (nn_weighted_adjective_noun_identity)
# IMPORTANCE                0.64   :   IMPORTANCE                0.48
# COMPLEXITY                0.29   :   position                  0.41
# duration                  0.20   :   size                      0.30
# normality                 0.19   :   COMPLEXITY                0.26
# crisis                    0.19   :   depth                     0.20
#
# LARGE_NUMBER                     :   VAST_AMOUNT                   (nn_weighted_adjective_noun_identity)
# SIZE                      0.75   :   SIZE                      0.67
# QUANTITY                  0.47   :   QUANTITY                  0.64
# WIDTH                     0.42   :   WIDTH                     0.45
# COMPLEXITY                0.33   :   COMPLEXITY                0.35
# weight                    0.31   :   depth                     0.34
#
# FEDERAL_ASSEMBLY                 :   NATIONAL_GOVERNMENT           (nn_weighted_adjective_noun_identity)
# SIZE                      0.16   :   importance                0.27
# speed                     0.15   :   SIZE                      0.23
# position                  0.14   :   crisis                    0.22
# DISTANCE                  0.13   :   quantity                  0.19
# weight                    0.13   :   DISTANCE                  0.18
#
# IMPORTANT_PART                   :   SIGNIFICANT_ROLE              (nn_weighted_adjective_noun_identity)
# IMPORTANCE                0.64   :   IMPORTANCE                0.48
# COMPLEXITY                0.29   :   position                  0.41
# duration                  0.20   :   size                      0.30
# normality                 0.19   :   COMPLEXITY                0.26
# crisis                    0.19   :   depth                     0.20
#
# CERTAIN_CIRCUMSTANCE             :   PARTICULAR_CASE               (nn_weighted_adjective_noun_identity)
# REGULARITY                0.30   :   importance                0.41
# normality                 0.25   :   size                      0.32
# position                  0.25   :   complexity                0.31
# reality                   0.24   :   duration                  0.30
# crisis                    0.20   :   REGULARITY                0.26
#
# IMPORTANT_PART                   :   SIGNIFICANT_ROLE              (nn_weighted_adjective_noun_identity)
# IMPORTANCE                0.64   :   IMPORTANCE                0.48
# COMPLEXITY                0.29   :   position                  0.41
# duration                  0.20   :   size                      0.30
# normality                 0.19   :   COMPLEXITY                0.26
# crisis                    0.19   :   depth                     0.20
#
# LARGE_NUMBER                     :   VAST_AMOUNT                   (nn_weighted_adjective_noun_identity)
# SIZE                      0.75   :   SIZE                      0.67
# QUANTITY                  0.47   :   QUANTITY                  0.64
# WIDTH                     0.42   :   WIDTH                     0.45
# COMPLEXITY                0.33   :   COMPLEXITY                0.35
# weight                    0.31   :   depth                     0.34
#
# GENERAL_PRINCIPLE                :   BASIC_RULE                    (nn_weighted_adjective_noun_identity)
# IMPORTANCE                0.36   :   IMPORTANCE                0.38
# NORMALITY                 0.25   :   NORMALITY                 0.35
# position                  0.23   :   COMPLEXITY                0.27
# REALITY                   0.21   :   REALITY                   0.26
# COMPLEXITY                0.16   :   consistency               0.25
#
# DIFFERENT_KIND                   :   VARIOUS_FORM                  (nn_weighted_adjective_noun_identity)
# CONSISTENCY               0.34   :   regularity                0.29
# complexity                0.33   :   CONSISTENCY               0.26
# NORMALITY                 0.33   :   NORMALITY                 0.26
# reality                   0.31   :   importance                0.23
# texture                   0.28   :   distance                  0.21
#
# VAST_AMOUNT                      :   LARGE_QUANTITY                (nn_weighted_adjective_noun_identity)
# SIZE                      0.67   :   QUANTITY                  0.83
# QUANTITY                  0.64   :   SIZE                      0.70
# WIDTH                     0.45   :   WIDTH                     0.45
# COMPLEXITY                0.35   :   COMPLEXITY                0.33
# DEPTH                     0.34   :   DEPTH                     0.29
#
# VAST_AMOUNT                      :   LARGE_QUANTITY                (nn_weighted_adjective_noun_identity)
# SIZE                      0.67   :   QUANTITY                  0.83
# QUANTITY                  0.64   :   SIZE                      0.70
# WIDTH                     0.45   :   WIDTH                     0.45
# COMPLEXITY                0.35   :   COMPLEXITY                0.33
# DEPTH                     0.34   :   DEPTH                     0.29
#
# SOCIAL_EVENT                     :   SPECIAL_CIRCUMSTANCE          (nn_weighted_adjective_noun_identity)
# IMPORTANCE                0.32   :   regularity                0.22
# crisis                    0.24   :   IMPORTANCE                0.20
# reality                   0.21   :   position                  0.19
# AGE                       0.20   :   complexity                0.19
# duration                  0.16   :   AGE                       0.18
#
# LARGE_NUMBER                     :   VAST_AMOUNT                   (nn_weighted_adjective_noun_identity)
# SIZE                      0.75   :   SIZE                      0.67
# QUANTITY                  0.47   :   QUANTITY                  0.64
# WIDTH                     0.42   :   WIDTH                     0.45
# COMPLEXITY                0.33   :   COMPLEXITY                0.35
# weight                    0.31   :   depth                     0.34
#
# GENERAL_PRINCIPLE                :   BASIC_RULE                    (nn_weighted_adjective_noun_identity)
# IMPORTANCE                0.36   :   IMPORTANCE                0.38
# NORMALITY                 0.25   :   NORMALITY                 0.35
# position                  0.23   :   COMPLEXITY                0.27
# REALITY                   0.21   :   REALITY                   0.26
# COMPLEXITY                0.16   :   consistency               0.25
#
# FEDERAL_ASSEMBLY                 :   NATIONAL_GOVERNMENT           (nn_weighted_adjective_noun_identity)
# SIZE                      0.16   :   importance                0.27
# speed                     0.15   :   SIZE                      0.23
# position                  0.14   :   crisis                    0.22
# DISTANCE                  0.13   :   quantity                  0.19
# weight                    0.13   :   DISTANCE                  0.18
#
# VAST_AMOUNT                      :   LARGE_QUANTITY                (nn_weighted_adjective_noun_identity)
# SIZE                      0.67   :   QUANTITY                  0.83
# QUANTITY                  0.64   :   SIZE                      0.70
# WIDTH                     0.45   :   WIDTH                     0.45
# COMPLEXITY                0.35   :   COMPLEXITY                0.33
# DEPTH                     0.34   :   DEPTH                     0.29
#
# SPECIAL_CIRCUMSTANCE             :   PARTICULAR_CASE               (nn_weighted_adjective_noun_identity)
# REGULARITY                0.22   :   IMPORTANCE                0.41
# IMPORTANCE                0.20   :   size                      0.32
# position                  0.19   :   COMPLEXITY                0.31
# COMPLEXITY                0.19   :   duration                  0.30
# age                       0.18   :   REGULARITY                0.26
#
# VAST_AMOUNT                      :   LARGE_QUANTITY                (nn_weighted_adjective_noun_identity)
# SIZE                      0.67   :   QUANTITY                  0.83
# QUANTITY                  0.64   :   SIZE                      0.70
# WIDTH                     0.45   :   WIDTH                     0.45
# COMPLEXITY                0.35   :   COMPLEXITY                0.33
# DEPTH                     0.34   :   DEPTH                     0.29
#
# LARGE_NUMBER                     :   VAST_AMOUNT                   (nn_weighted_adjective_noun_identity)
# SIZE                      0.75   :   SIZE                      0.67
# QUANTITY                  0.47   :   QUANTITY                  0.64
# WIDTH                     0.42   :   WIDTH                     0.45
# COMPLEXITY                0.33   :   COMPLEXITY                0.35
# weight                    0.31   :   depth                     0.34
#
# OLD_PERSON                       :   ELDERLY_LADY                  (nn_weighted_adjective_noun_identity)
# AGE                       0.59   :   AGE                       0.33
# COMPLEXION                0.26   :   COMPLEXION                0.16
# duration                  0.20   :   SIZE                      0.15
# distance                  0.17   :   crisis                    0.14
# SIZE                      0.17   :   regularity                0.12
#
# DIFFERENT_KIND                   :   VARIOUS_FORM                  (nn_weighted_adjective_noun_identity)
# CONSISTENCY               0.34   :   regularity                0.29
# complexity                0.33   :   CONSISTENCY               0.26
# NORMALITY                 0.33   :   NORMALITY                 0.26
# reality                   0.31   :   importance                0.23
# texture                   0.28   :   distance                  0.21
#
# LARGE_QUANTITY                   :   GREAT_MAJORITY                (nn_weighted_adjective_noun_identity)
# QUANTITY                  0.83   :   SIZE                      0.44
# SIZE                      0.70   :   importance                0.41
# WIDTH                     0.45   :   WIDTH                     0.28
# complexity                0.33   :   QUANTITY                  0.26
# depth                     0.29   :   weight                    0.25
#
# MAJOR_ISSUE                      :   AMERICAN_COUNTRY              (nn_weighted_adjective_noun_identity)
# SIZE                      0.43   :   reality                   0.24
# importance                0.41   :   color                     0.22
# complexity                0.32   :   complexion                0.19
# duration                  0.30   :   SIZE                      0.16
# WIDTH                     0.26   :   WIDTH                     0.12
#
# LARGE_NUMBER                     :   GREAT_MAJORITY                (nn_weighted_adjective_noun_identity)
# SIZE                      0.75   :   SIZE                      0.44
# QUANTITY                  0.47   :   importance                0.41
# WIDTH                     0.42   :   WIDTH                     0.28
# complexity                0.33   :   QUANTITY                  0.26
# WEIGHT                    0.31   :   WEIGHT                    0.25
#
# CERTAIN_CIRCUMSTANCE             :   PARTICULAR_CASE               (nn_weighted_adjective_noun_identity)
# REGULARITY                0.30   :   importance                0.41
# normality                 0.25   :   size                      0.32
# position                  0.25   :   complexity                0.31
# reality                   0.24   :   duration                  0.30
# crisis                    0.20   :   REGULARITY                0.26
#
# IMPORTANT_PART                   :   SIGNIFICANT_ROLE              (nn_weighted_adjective_noun_identity)
# IMPORTANCE                0.64   :   IMPORTANCE                0.48
# COMPLEXITY                0.29   :   position                  0.41
# duration                  0.20   :   size                      0.30
# normality                 0.19   :   COMPLEXITY                0.26
# crisis                    0.19   :   depth                     0.20
#
# LARGE_NUMBER                     :   GREAT_MAJORITY                (nn_weighted_adjective_noun_identity)
# SIZE                      0.75   :   SIZE                      0.44
# QUANTITY                  0.47   :   importance                0.41
# WIDTH                     0.42   :   WIDTH                     0.28
# complexity                0.33   :   QUANTITY                  0.26
# WEIGHT                    0.31   :   WEIGHT                    0.25
#
# CERTAIN_CIRCUMSTANCE             :   PARTICULAR_CASE               (nn_weighted_adjective_noun_identity)
# REGULARITY                0.30   :   importance                0.41
# normality                 0.25   :   size                      0.32
# position                  0.25   :   complexity                0.31
# reality                   0.24   :   duration                  0.30
# crisis                    0.20   :   REGULARITY                0.26
#
# IMPORTANT_PART                   :   SIGNIFICANT_ROLE              (nn_weighted_adjective_noun_identity)
# IMPORTANCE                0.64   :   IMPORTANCE                0.48
# COMPLEXITY                0.29   :   position                  0.41
# duration                  0.20   :   size                      0.30
# normality                 0.19   :   COMPLEXITY                0.26
# crisis                    0.19   :   depth                     0.20
#
# IMPORTANT_PART                   :   SIGNIFICANT_ROLE              (nn_weighted_adjective_noun_identity)
# IMPORTANCE                0.64   :   IMPORTANCE                0.48
# COMPLEXITY                0.29   :   position                  0.41
# duration                  0.20   :   size                      0.30
# normality                 0.19   :   COMPLEXITY                0.26
# crisis                    0.19   :   depth                     0.20
#
# LARGE_NUMBER                     :   VAST_AMOUNT                   (nn_weighted_adjective_noun_identity)
# SIZE                      0.75   :   SIZE                      0.67
# QUANTITY                  0.47   :   QUANTITY                  0.64
# WIDTH                     0.42   :   WIDTH                     0.45
# COMPLEXITY                0.33   :   COMPLEXITY                0.35
# weight                    0.31   :   depth                     0.34
#
# LARGE_NUMBER                     :   VAST_AMOUNT                   (nn_weighted_adjective_noun_identity)
# SIZE                      0.75   :   SIZE                      0.67
# QUANTITY                  0.47   :   QUANTITY                  0.64
# WIDTH                     0.42   :   WIDTH                     0.45
# COMPLEXITY                0.33   :   COMPLEXITY                0.35
# weight                    0.31   :   depth                     0.34
#
# VAST_AMOUNT                      :   LARGE_QUANTITY                (nn_weighted_adjective_noun_identity)
# SIZE                      0.67   :   QUANTITY                  0.83
# QUANTITY                  0.64   :   SIZE                      0.70
# WIDTH                     0.45   :   WIDTH                     0.45
# COMPLEXITY                0.35   :   COMPLEXITY                0.33
# DEPTH                     0.34   :   DEPTH                     0.29
#
# VAST_AMOUNT                      :   LARGE_QUANTITY                (nn_weighted_adjective_noun_identity)
# SIZE                      0.67   :   QUANTITY                  0.83
# QUANTITY                  0.64   :   SIZE                      0.70
# WIDTH                     0.45   :   WIDTH                     0.45
# COMPLEXITY                0.35   :   COMPLEXITY                0.33
# DEPTH                     0.34   :   DEPTH                     0.29
#
# NEW_INFORMATION                  :   FURTHER_EVIDENCE              (nn_weighted_adjective_noun_identity)
# IMPORTANCE                0.30   :   DEPTH                     0.28
# complexity                0.28   :   quantity                  0.24
# age                       0.25   :   IMPORTANCE                0.21
# width                     0.24   :   consistency               0.19
# DEPTH                     0.24   :   weight                    0.19
#
# VAST_AMOUNT                      :   LARGE_QUANTITY                (nn_weighted_adjective_noun_identity)
# SIZE                      0.67   :   QUANTITY                  0.83
# QUANTITY                  0.64   :   SIZE                      0.70
# WIDTH                     0.45   :   WIDTH                     0.45
# COMPLEXITY                0.35   :   COMPLEXITY                0.33
# DEPTH                     0.34   :   DEPTH                     0.29
#
# LARGE_NUMBER                     :   VAST_AMOUNT                   (nn_weighted_adjective_noun_identity)
# SIZE                      0.75   :   SIZE                      0.67
# QUANTITY                  0.47   :   QUANTITY                  0.64
# WIDTH                     0.42   :   WIDTH                     0.45
# COMPLEXITY                0.33   :   COMPLEXITY                0.35
# weight                    0.31   :   depth                     0.34
#
# LARGE_QUANTITY                   :   GREAT_MAJORITY                (nn_weighted_adjective_noun_identity)
# QUANTITY                  0.83   :   SIZE                      0.44
# SIZE                      0.70   :   importance                0.41
# WIDTH                     0.45   :   WIDTH                     0.28
# complexity                0.33   :   QUANTITY                  0.26
# depth                     0.29   :   weight                    0.25
#
# LARGE_NUMBER                     :   GREAT_MAJORITY                (nn_weighted_adjective_noun_identity)
# SIZE                      0.75   :   SIZE                      0.44
# QUANTITY                  0.47   :   importance                0.41
# WIDTH                     0.42   :   WIDTH                     0.28
# complexity                0.33   :   QUANTITY                  0.26
# WEIGHT                    0.31   :   WEIGHT                    0.25
#
# LARGE_NUMBER                     :   VAST_AMOUNT                   (nn_weighted_adjective_noun_identity)
# SIZE                      0.75   :   SIZE                      0.67
# QUANTITY                  0.47   :   QUANTITY                  0.64
# WIDTH                     0.42   :   WIDTH                     0.45
# COMPLEXITY                0.33   :   COMPLEXITY                0.35
# weight                    0.31   :   depth                     0.34
#
# OLD_PERSON                       :   ELDERLY_LADY                  (nn_weighted_adjective_noun_identity)
# AGE                       0.59   :   AGE                       0.33
# COMPLEXION                0.26   :   COMPLEXION                0.16
# duration                  0.20   :   SIZE                      0.15
# distance                  0.17   :   crisis                    0.14
# SIZE                      0.17   :   regularity                0.12
#
# EARLIER_WORK                     :   EARLY_STAGE                   (nn_weighted_adjective_noun_identity)
# importance                0.33   :   age                       0.31
# regularity                0.26   :   DURATION                  0.23
# normality                 0.24   :   crisis                    0.23
# DURATION                  0.21   :   speed                     0.22
# POSITION                  0.20   :   POSITION                  0.22
#
# GENERAL_PRINCIPLE                :   BASIC_RULE                    (nn_weighted_adjective_noun_identity)
# IMPORTANCE                0.36   :   IMPORTANCE                0.38
# NORMALITY                 0.25   :   NORMALITY                 0.35
# position                  0.23   :   COMPLEXITY                0.27
# REALITY                   0.21   :   REALITY                   0.26
# COMPLEXITY                0.16   :   consistency               0.25
#
# LARGE_QUANTITY                   :   GREAT_MAJORITY                (nn_weighted_adjective_noun_identity)
# QUANTITY                  0.83   :   SIZE                      0.44
# SIZE                      0.70   :   importance                0.41
# WIDTH                     0.45   :   WIDTH                     0.28
# complexity                0.33   :   QUANTITY                  0.26
# depth                     0.29   :   weight                    0.25
#
# NEW_LIFE                         :   EARLY_AGE                     (nn_weighted_adjective_noun_identity)
# DURATION                  0.39   :   AGE                       0.81
# AGE                       0.31   :   DURATION                  0.25
# size                      0.29   :   temperature               0.25
# reality                   0.28   :   speed                     0.22
# complexity                0.24   :   distance                  0.21
#
# GENERAL_PRINCIPLE                :   BASIC_RULE                    (nn_weighted_adjective_noun_identity)
# IMPORTANCE                0.36   :   IMPORTANCE                0.38
# NORMALITY                 0.25   :   NORMALITY                 0.35
# position                  0.23   :   COMPLEXITY                0.27
# REALITY                   0.21   :   REALITY                   0.26
# COMPLEXITY                0.16   :   consistency               0.25
#
# DIFFERENT_KIND                   :   VARIOUS_FORM                  (nn_weighted_adjective_noun_identity)
# CONSISTENCY               0.34   :   regularity                0.29
# complexity                0.33   :   CONSISTENCY               0.26
# NORMALITY                 0.33   :   NORMALITY                 0.26
# reality                   0.31   :   importance                0.23
# texture                   0.28   :   distance                  0.21
#
# CERTAIN_CIRCUMSTANCE             :   PARTICULAR_CASE               (nn_weighted_adjective_noun_identity)
# REGULARITY                0.30   :   importance                0.41
# normality                 0.25   :   size                      0.32
# position                  0.25   :   complexity                0.31
# reality                   0.24   :   duration                  0.30
# crisis                    0.20   :   REGULARITY                0.26
#
# IMPORTANT_PART                   :   SIGNIFICANT_ROLE              (nn_weighted_adjective_noun_identity)
# IMPORTANCE                0.64   :   IMPORTANCE                0.48
# COMPLEXITY                0.29   :   position                  0.41
# duration                  0.20   :   size                      0.30
# normality                 0.19   :   COMPLEXITY                0.26
# crisis                    0.19   :   depth                     0.20
#
# VAST_AMOUNT                      :   LARGE_QUANTITY                (nn_weighted_adjective_noun_identity)
# SIZE                      0.67   :   QUANTITY                  0.83
# QUANTITY                  0.64   :   SIZE                      0.70
# WIDTH                     0.45   :   WIDTH                     0.45
# COMPLEXITY                0.35   :   COMPLEXITY                0.33
# DEPTH                     0.34   :   DEPTH                     0.29
#
# LARGE_NUMBER                     :   VAST_AMOUNT                   (nn_weighted_adjective_noun_identity)
# SIZE                      0.75   :   SIZE                      0.67
# QUANTITY                  0.47   :   QUANTITY                  0.64
# WIDTH                     0.42   :   WIDTH                     0.45
# COMPLEXITY                0.33   :   COMPLEXITY                0.35
# weight                    0.31   :   depth                     0.34
#
# GENERAL_PRINCIPLE                :   BASIC_RULE                    (nn_weighted_adjective_noun_identity)
# IMPORTANCE                0.36   :   IMPORTANCE                0.38
# NORMALITY                 0.25   :   NORMALITY                 0.35
# position                  0.23   :   COMPLEXITY                0.27
# REALITY                   0.21   :   REALITY                   0.26
# COMPLEXITY                0.16   :   consistency               0.25
#
# DIFFERENT_KIND                   :   VARIOUS_FORM                  (nn_weighted_adjective_noun_identity)
# CONSISTENCY               0.34   :   regularity                0.29
# complexity                0.33   :   CONSISTENCY               0.26
# NORMALITY                 0.33   :   NORMALITY                 0.26
# reality                   0.31   :   importance                0.23
# texture                   0.28   :   distance                  0.21
#
# FEDERAL_ASSEMBLY                 :   NATIONAL_GOVERNMENT           (nn_weighted_adjective_noun_identity)
# SIZE                      0.16   :   importance                0.27
# speed                     0.15   :   SIZE                      0.23
# position                  0.14   :   crisis                    0.22
# DISTANCE                  0.13   :   quantity                  0.19
# weight                    0.13   :   DISTANCE                  0.18
#
# LARGE_NUMBER                     :   VAST_AMOUNT                   (nn_weighted_adjective_noun_identity)
# SIZE                      0.75   :   SIZE                      0.67
# QUANTITY                  0.47   :   QUANTITY                  0.64
# WIDTH                     0.42   :   WIDTH                     0.45
# COMPLEXITY                0.33   :   COMPLEXITY                0.35
# weight                    0.31   :   depth                     0.34
#
# OLD_PERSON                       :   ELDERLY_LADY                  (nn_weighted_adjective_noun_identity)
# AGE                       0.59   :   AGE                       0.33
# COMPLEXION                0.26   :   COMPLEXION                0.16
# duration                  0.20   :   SIZE                      0.15
# distance                  0.17   :   crisis                    0.14
# SIZE                      0.17   :   regularity                0.12
#
# NEW_INFORMATION                  :   FURTHER_EVIDENCE              (nn_weighted_adjective_noun_identity)
# IMPORTANCE                0.30   :   DEPTH                     0.28
# complexity                0.28   :   quantity                  0.24
# age                       0.25   :   IMPORTANCE                0.21
# width                     0.24   :   consistency               0.19
# DEPTH                     0.24   :   weight                    0.19
#
# SPECIAL_CIRCUMSTANCE             :   PARTICULAR_CASE               (nn_weighted_adjective_noun_identity)
# REGULARITY                0.22   :   IMPORTANCE                0.41
# IMPORTANCE                0.20   :   size                      0.32
# position                  0.19   :   COMPLEXITY                0.31
# COMPLEXITY                0.19   :   duration                  0.30
# age                       0.18   :   REGULARITY                0.26
#
# VAST_AMOUNT                      :   LARGE_QUANTITY                (nn_weighted_adjective_noun_identity)
# SIZE                      0.67   :   QUANTITY                  0.83
# QUANTITY                  0.64   :   SIZE                      0.70
# WIDTH                     0.45   :   WIDTH                     0.45
# COMPLEXITY                0.35   :   COMPLEXITY                0.33
# DEPTH                     0.34   :   DEPTH                     0.29
#
# VAST_AMOUNT                      :   LARGE_QUANTITY                (nn_weighted_adjective_noun_identity)
# SIZE                      0.67   :   QUANTITY                  0.83
# QUANTITY                  0.64   :   SIZE                      0.70
# WIDTH                     0.45   :   WIDTH                     0.45
# COMPLEXITY                0.35   :   COMPLEXITY                0.33
# DEPTH                     0.34   :   DEPTH                     0.29
#
# NEW_INFORMATION                  :   FURTHER_EVIDENCE              (nn_weighted_adjective_noun_identity)
# IMPORTANCE                0.30   :   DEPTH                     0.28
# complexity                0.28   :   quantity                  0.24
# age                       0.25   :   IMPORTANCE                0.21
# width                     0.24   :   consistency               0.19
# DEPTH                     0.24   :   weight                    0.19
#
# VAST_AMOUNT                      :   LARGE_QUANTITY                (nn_weighted_adjective_noun_identity)
# SIZE                      0.67   :   QUANTITY                  0.83
# QUANTITY                  0.64   :   SIZE                      0.70
# WIDTH                     0.45   :   WIDTH                     0.45
# COMPLEXITY                0.35   :   COMPLEXITY                0.33
# DEPTH                     0.34   :   DEPTH                     0.29
#
# LARGE_NUMBER                     :   VAST_AMOUNT                   (nn_weighted_adjective_noun_identity)
# SIZE                      0.75   :   SIZE                      0.67
# QUANTITY                  0.47   :   QUANTITY                  0.64
# WIDTH                     0.42   :   WIDTH                     0.45
# COMPLEXITY                0.33   :   COMPLEXITY                0.35
# weight                    0.31   :   depth                     0.34
#
# OLD_PERSON                       :   ELDERLY_LADY                  (nn_weighted_adjective_noun_identity)
# AGE                       0.59   :   AGE                       0.33
# COMPLEXION                0.26   :   COMPLEXION                0.16
# duration                  0.20   :   SIZE                      0.15
# distance                  0.17   :   crisis                    0.14
# SIZE                      0.17   :   regularity                0.12
#
# DIFFERENT_KIND                   :   VARIOUS_FORM                  (nn_weighted_adjective_noun_identity)
# CONSISTENCY               0.34   :   regularity                0.29
# complexity                0.33   :   CONSISTENCY               0.26
# NORMALITY                 0.33   :   NORMALITY                 0.26
# reality                   0.31   :   importance                0.23
# texture                   0.28   :   distance                  0.21
#
# GENERAL_PRINCIPLE                :   BASIC_RULE                    (nn_weighted_adjective_noun_identity)
# IMPORTANCE                0.36   :   IMPORTANCE                0.38
# NORMALITY                 0.25   :   NORMALITY                 0.35
# position                  0.23   :   COMPLEXITY                0.27
# REALITY                   0.21   :   REALITY                   0.26
# COMPLEXITY                0.16   :   consistency               0.25
#
# LARGE_NUMBER                     :   VAST_AMOUNT                   (nn_weighted_adjective_noun_identity)
# SIZE                      0.75   :   SIZE                      0.67
# QUANTITY                  0.47   :   QUANTITY                  0.64
# WIDTH                     0.42   :   WIDTH                     0.45
# COMPLEXITY                0.33   :   COMPLEXITY                0.35
# weight                    0.31   :   depth                     0.34
#
# GENERAL_PRINCIPLE                :   BASIC_RULE                    (nn_weighted_adjective_noun_identity)
# IMPORTANCE                0.36   :   IMPORTANCE                0.38
# NORMALITY                 0.25   :   NORMALITY                 0.35
# position                  0.23   :   COMPLEXITY                0.27
# REALITY                   0.21   :   REALITY                   0.26
# COMPLEXITY                0.16   :   consistency               0.25
#
# LARGE_QUANTITY                   :   GREAT_MAJORITY                (nn_weighted_adjective_noun_identity)
# QUANTITY                  0.83   :   SIZE                      0.44
# SIZE                      0.70   :   importance                0.41
# WIDTH                     0.45   :   WIDTH                     0.28
# complexity                0.33   :   QUANTITY                  0.26
# depth                     0.29   :   weight                    0.25
#
# IMPORTANT_PART                   :   SIGNIFICANT_ROLE              (nn_weighted_adjective_noun_identity)
# IMPORTANCE                0.64   :   IMPORTANCE                0.48
# COMPLEXITY                0.29   :   position                  0.41
# duration                  0.20   :   size                      0.30
# normality                 0.19   :   COMPLEXITY                0.26
# crisis                    0.19   :   depth                     0.20
#
# NEW_INFORMATION                  :   FURTHER_EVIDENCE              (nn_weighted_adjective_noun_identity)
# IMPORTANCE                0.30   :   DEPTH                     0.28
# complexity                0.28   :   quantity                  0.24
# age                       0.25   :   IMPORTANCE                0.21
# width                     0.24   :   consistency               0.19
# DEPTH                     0.24   :   weight                    0.19
#
# VAST_AMOUNT                      :   LARGE_QUANTITY                (nn_weighted_adjective_noun_identity)
# SIZE                      0.67   :   QUANTITY                  0.83
# QUANTITY                  0.64   :   SIZE                      0.70
# WIDTH                     0.45   :   WIDTH                     0.45
# COMPLEXITY                0.35   :   COMPLEXITY                0.33
# DEPTH                     0.34   :   DEPTH                     0.29
#
# LARGE_NUMBER                     :   GREAT_MAJORITY                (nn_weighted_adjective_noun_identity)
# SIZE                      0.75   :   SIZE                      0.44
# QUANTITY                  0.47   :   importance                0.41
# WIDTH                     0.42   :   WIDTH                     0.28
# complexity                0.33   :   QUANTITY                  0.26
# WEIGHT                    0.31   :   WEIGHT                    0.25
#
# SPECIAL_CIRCUMSTANCE             :   PARTICULAR_CASE               (nn_weighted_adjective_noun_identity)
# REGULARITY                0.22   :   IMPORTANCE                0.41
# IMPORTANCE                0.20   :   size                      0.32
# position                  0.19   :   COMPLEXITY                0.31
# COMPLEXITY                0.19   :   duration                  0.30
# age                       0.18   :   REGULARITY                0.26
#
# INDUSTRIAL_AREA                  :   WHOLE_COUNTRY                 (nn_weighted_adjective_noun_identity)
# WIDTH                     0.32   :   SIZE                      0.38
# SIZE                      0.26   :   WIDTH                     0.33
# temperature               0.25   :   complexity                0.29
# quantity                  0.23   :   reality                   0.29
# speed                     0.23   :   duration                  0.26
#
# SOCIAL_ACTIVITY                  :   ECONOMIC_CONDITION            (nn_weighted_adjective_noun_identity)
# NORMALITY                 0.31   :   crisis                    0.36
# IMPORTANCE                0.26   :   position                  0.30
# duration                  0.26   :   IMPORTANCE                0.25
# regularity                0.19   :   NORMALITY                 0.20
# complexity                0.19   :   reality                   0.20
#
# GOOD_PLACE                       :   HIGH_POINT                    (nn_weighted_adjective_noun_identity)
# importance                0.36   :   speed                     0.34
# CONSISTENCY               0.34   :   CONSISTENCY               0.33
# position                  0.32   :   DISTANCE                  0.32
# DISTANCE                  0.25   :   width                     0.25
# texture                   0.24   :   quantity                  0.25
#
# NEW_INFORMATION                  :   FURTHER_EVIDENCE              (nn_weighted_adjective_noun_identity)
# IMPORTANCE                0.30   :   DEPTH                     0.28
# complexity                0.28   :   quantity                  0.24
# age                       0.25   :   IMPORTANCE                0.21
# width                     0.24   :   consistency               0.19
# DEPTH                     0.24   :   weight                    0.19
#
# DIFFERENT_PART                   :   VARIOUS_FORM                  (nn_weighted_adjective_noun_identity)
# complexity                0.35   :   regularity                0.29
# IMPORTANCE                0.28   :   consistency               0.26
# NORMALITY                 0.28   :   NORMALITY                 0.26
# reality                   0.24   :   IMPORTANCE                0.23
# size                      0.24   :   distance                  0.21
#
# ECONOMIC_PROBLEM                 :   PRACTICAL_DIFFICULTY          (nn_weighted_adjective_noun_identity)
# crisis                    0.54   :   COMPLEXITY                0.56
# IMPORTANCE                0.33   :   IMPORTANCE                0.42
# COMPLEXITY                0.29   :   weight                    0.26
# REALITY                   0.29   :   REALITY                   0.26
# size                      0.26   :   depth                     0.25
#
# EFFECTIVE_WAY                    :   IMPORTANT_PART                (nn_weighted_adjective_noun_identity)
# consistency               0.27   :   IMPORTANCE                0.64
# position                  0.25   :   COMPLEXITY                0.29
# COMPLEXITY                0.25   :   duration                  0.20
# IMPORTANCE                0.25   :   normality                 0.19
# regularity                0.23   :   crisis                    0.19
#
# NEW_SITUATION                    :   DIFFERENT_KIND                (nn_weighted_adjective_noun_identity)
# position                  0.39   :   consistency               0.34
# crisis                    0.34   :   COMPLEXITY                0.33
# REALITY                   0.28   :   normality                 0.33
# COMPLEXITY                0.27   :   REALITY                   0.31
# speed                     0.26   :   texture                   0.28
#
# LARGE_NUMBER                     :   GREAT_MAJORITY                (nn_weighted_adjective_noun_identity)
# SIZE                      0.75   :   SIZE                      0.44
# QUANTITY                  0.47   :   importance                0.41
# WIDTH                     0.42   :   WIDTH                     0.28
# complexity                0.33   :   QUANTITY                  0.26
# WEIGHT                    0.31   :   WEIGHT                    0.25
#
# CERTAIN_CIRCUMSTANCE             :   PARTICULAR_CASE               (nn_weighted_adjective_noun_identity)
# REGULARITY                0.30   :   importance                0.41
# normality                 0.25   :   size                      0.32
# position                  0.25   :   complexity                0.31
# reality                   0.24   :   duration                  0.30
# crisis                    0.20   :   REGULARITY                0.26
#
# VAST_AMOUNT                      :   HIGH_PRICE                    (nn_weighted_adjective_noun_identity)
# SIZE                      0.67   :   QUANTITY                  0.45
# QUANTITY                  0.64   :   SIZE                      0.34
# WIDTH                     0.45   :   temperature               0.32
# complexity                0.35   :   speed                     0.30
# depth                     0.34   :   WIDTH                     0.29
#
# NEW_LAW                          :   BASIC_RULE                    (nn_weighted_adjective_noun_identity)
# size                      0.29   :   importance                0.38
# age                       0.29   :   normality                 0.35
# width                     0.23   :   complexity                0.27
# REALITY                   0.19   :   REALITY                   0.26
# speed                     0.18   :   consistency               0.25
#
# CENTRAL_AUTHORITY                :   LOCAL_OFFICE                  (nn_weighted_adjective_noun_identity)
# IMPORTANCE                0.45   :   size                      0.30
# POSITION                  0.26   :   distance                  0.20
# consistency               0.24   :   POSITION                  0.20
# width                     0.23   :   IMPORTANCE                0.18
# normality                 0.23   :   temperature               0.18
#
# LARGE_NUMBER                     :   GREAT_MAJORITY                (nn_weighted_adjective_noun_identity)
# SIZE                      0.75   :   SIZE                      0.44
# QUANTITY                  0.47   :   importance                0.41
# WIDTH                     0.42   :   WIDTH                     0.28
# complexity                0.33   :   QUANTITY                  0.26
# WEIGHT                    0.31   :   WEIGHT                    0.25
#
# OLD_PERSON                       :   ELDERLY_LADY                  (nn_weighted_adjective_noun_identity)
# AGE                       0.59   :   AGE                       0.33
# COMPLEXION                0.26   :   COMPLEXION                0.16
# duration                  0.20   :   SIZE                      0.15
# distance                  0.17   :   crisis                    0.14
# SIZE                      0.17   :   regularity                0.12
#
# GENERAL_PRINCIPLE                :   BASIC_RULE                    (nn_weighted_adjective_noun_identity)
# IMPORTANCE                0.36   :   IMPORTANCE                0.38
# NORMALITY                 0.25   :   NORMALITY                 0.35
# position                  0.23   :   COMPLEXITY                0.27
# REALITY                   0.21   :   REALITY                   0.26
# COMPLEXITY                0.16   :   consistency               0.25
#
# LARGE_NUMBER                     :   GREAT_MAJORITY                (nn_weighted_adjective_noun_identity)
# SIZE                      0.75   :   SIZE                      0.44
# QUANTITY                  0.47   :   importance                0.41
# WIDTH                     0.42   :   WIDTH                     0.28
# complexity                0.33   :   QUANTITY                  0.26
# WEIGHT                    0.31   :   WEIGHT                    0.25
#
# CERTAIN_CIRCUMSTANCE             :   PARTICULAR_CASE               (nn_weighted_adjective_noun_identity)
# REGULARITY                0.30   :   importance                0.41
# normality                 0.25   :   size                      0.32
# position                  0.25   :   complexity                0.31
# reality                   0.24   :   duration                  0.30
# crisis                    0.20   :   REGULARITY                0.26
#
# PREVIOUS_DAY                     :   EARLY_AGE                     (nn_weighted_adjective_noun_identity)
# DURATION                  0.39   :   AGE                       0.81
# regularity                0.29   :   DURATION                  0.25
# AGE                       0.26   :   TEMPERATURE               0.25
# light                     0.24   :   speed                     0.22
# TEMPERATURE               0.22   :   distance                  0.21
#
# LARGE_NUMBER                     :   VAST_AMOUNT                   (nn_weighted_adjective_noun_identity)
# SIZE                      0.75   :   SIZE                      0.67
# QUANTITY                  0.47   :   QUANTITY                  0.64
# WIDTH                     0.42   :   WIDTH                     0.45
# COMPLEXITY                0.33   :   COMPLEXITY                0.35
# weight                    0.31   :   depth                     0.34
#
# EFFECTIVE_WAY                    :   EFFICIENT_USE                 (nn_weighted_adjective_noun_identity)
# CONSISTENCY               0.27   :   speed                     0.38
# position                  0.25   :   CONSISTENCY               0.29
# COMPLEXITY                0.25   :   COMPLEXITY                0.28
# importance                0.25   :   REGULARITY                0.21
# REGULARITY                0.23   :   absorbency                0.20
#
# LARGE_NUMBER                     :   GREAT_MAJORITY                (nn_weighted_adjective_noun_identity)
# SIZE                      0.75   :   SIZE                      0.44
# QUANTITY                  0.47   :   importance                0.41
# WIDTH                     0.42   :   WIDTH                     0.28
# complexity                0.33   :   QUANTITY                  0.26
# WEIGHT                    0.31   :   WEIGHT                    0.25
#
# RIGHT_HAND                       :   LEFT_ARM                      (nn_weighted_adjective_noun_identity)
# POSITION                  0.84   :   POSITION                  0.57
# importance                0.28   :   distance                  0.30
# SPEED                     0.22   :   DEPTH                     0.22
# DEPTH                     0.22   :   SPEED                     0.22
# width                     0.20   :   size                      0.20
#
# OLD_PERSON                       :   ELDERLY_LADY                  (nn_weighted_adjective_noun_identity)
# AGE                       0.59   :   AGE                       0.33
# COMPLEXION                0.26   :   COMPLEXION                0.16
# duration                  0.20   :   SIZE                      0.15
# distance                  0.17   :   crisis                    0.14
# SIZE                      0.17   :   regularity                0.12
#
# FEDERAL_ASSEMBLY                 :   NATIONAL_GOVERNMENT           (nn_weighted_adjective_noun_identity)
# SIZE                      0.16   :   importance                0.27
# speed                     0.15   :   SIZE                      0.23
# position                  0.14   :   crisis                    0.22
# DISTANCE                  0.13   :   quantity                  0.19
# weight                    0.13   :   DISTANCE                  0.18
#
# BLACK_HAIR                       :   DARK_EYE                      (nn_weighted_adjective_noun_identity)
# COLOR                     0.73   :   LIGHT                     0.49
# COMPLEXION                0.48   :   COLOR                     0.44
# TEXTURE                   0.31   :   COMPLEXION                0.40
# size                      0.25   :   depth                     0.22
# LIGHT                     0.23   :   TEXTURE                   0.22
#
# SMALL_HOUSE                      :   LITTLE_ROOM                   (nn_weighted_adjective_noun_identity)
# SIZE                      0.83   :   SIZE                      0.51
# width                     0.46   :   temperature               0.40
# quantity                  0.34   :   depth                     0.38
# complexity                0.33   :   weight                    0.38
# age                       0.25   :   speed                     0.34
#
# AMERICAN_COUNTRY                 :   EUROPEAN_STATE                (nn_weighted_adjective_noun_identity)
# reality                   0.24   :   importance                0.17
# color                     0.22   :   quantity                  0.16
# complexion                0.19   :   consistency               0.14
# SIZE                      0.16   :   WIDTH                     0.14
# WIDTH                     0.12   :   SIZE                      0.12
#
# NEW_INFORMATION                  :   FURTHER_EVIDENCE              (nn_weighted_adjective_noun_identity)
# IMPORTANCE                0.30   :   DEPTH                     0.28
# complexity                0.28   :   quantity                  0.24
# age                       0.25   :   IMPORTANCE                0.21
# width                     0.24   :   consistency               0.19
# DEPTH                     0.24   :   weight                    0.19
#
# SPECIAL_CIRCUMSTANCE             :   PARTICULAR_CASE               (nn_weighted_adjective_noun_identity)
# REGULARITY                0.22   :   IMPORTANCE                0.41
# IMPORTANCE                0.20   :   size                      0.32
# position                  0.19   :   COMPLEXITY                0.31
# COMPLEXITY                0.19   :   duration                  0.30
# age                       0.18   :   REGULARITY                0.26
#
# DIFFERENT_PART                   :   VARIOUS_FORM                  (nn_weighted_adjective_noun_identity)
# complexity                0.35   :   regularity                0.29
# IMPORTANCE                0.28   :   consistency               0.26
# NORMALITY                 0.28   :   NORMALITY                 0.26
# reality                   0.24   :   IMPORTANCE                0.23
# size                      0.24   :   distance                  0.21
#
# SPECIAL_CIRCUMSTANCE             :   PARTICULAR_CASE               (nn_weighted_adjective_noun_identity)
# REGULARITY                0.22   :   IMPORTANCE                0.41
# IMPORTANCE                0.20   :   size                      0.32
# position                  0.19   :   COMPLEXITY                0.31
# COMPLEXITY                0.19   :   duration                  0.30
# age                       0.18   :   REGULARITY                0.26
#
# NEW_SITUATION                    :   DIFFERENT_KIND                (nn_weighted_adjective_noun_identity)
# position                  0.39   :   consistency               0.34
# crisis                    0.34   :   COMPLEXITY                0.33
# REALITY                   0.28   :   normality                 0.33
# COMPLEXITY                0.27   :   REALITY                   0.31
# speed                     0.26   :   texture                   0.28
#
# OLD_PERSON                       :   ELDERLY_LADY                  (nn_weighted_adjective_noun_identity)
# AGE                       0.59   :   AGE                       0.33
# COMPLEXION                0.26   :   COMPLEXION                0.16
# duration                  0.20   :   SIZE                      0.15
# distance                  0.17   :   crisis                    0.14
# SIZE                      0.17   :   regularity                0.12
#
# EARLIER_WORK                     :   EARLY_STAGE                   (nn_weighted_adjective_noun_identity)
# importance                0.33   :   age                       0.31
# regularity                0.26   :   DURATION                  0.23
# normality                 0.24   :   crisis                    0.23
# DURATION                  0.21   :   speed                     0.22
# POSITION                  0.20   :   POSITION                  0.22
#
# DIFFERENT_KIND                   :   VARIOUS_FORM                  (nn_weighted_adjective_noun_identity)
# CONSISTENCY               0.34   :   regularity                0.29
# complexity                0.33   :   CONSISTENCY               0.26
# NORMALITY                 0.33   :   NORMALITY                 0.26
# reality                   0.31   :   importance                0.23
# texture                   0.28   :   distance                  0.21
#
# EFFECTIVE_WAY                    :   EFFICIENT_USE                 (nn_weighted_adjective_noun_identity)
# CONSISTENCY               0.27   :   speed                     0.38
# position                  0.25   :   CONSISTENCY               0.29
# COMPLEXITY                0.25   :   COMPLEXITY                0.28
# importance                0.25   :   REGULARITY                0.21
# REGULARITY                0.23   :   absorbency                0.20
#
# MAJOR_ISSUE                      :   AMERICAN_COUNTRY              (nn_weighted_adjective_noun_identity)
# SIZE                      0.43   :   reality                   0.24
# importance                0.41   :   color                     0.22
# complexity                0.32   :   complexion                0.19
# duration                  0.30   :   SIZE                      0.16
# WIDTH                     0.26   :   WIDTH                     0.12
#
# LARGE_QUANTITY                   :   GREAT_MAJORITY                (nn_weighted_adjective_noun_identity)
# QUANTITY                  0.83   :   SIZE                      0.44
# SIZE                      0.70   :   importance                0.41
# WIDTH                     0.45   :   WIDTH                     0.28
# complexity                0.33   :   QUANTITY                  0.26
# depth                     0.29   :   weight                    0.25
#
# BLACK_HAIR                       :   DARK_EYE                      (nn_weighted_adjective_noun_identity)
# COLOR                     0.73   :   LIGHT                     0.49
# COMPLEXION                0.48   :   COLOR                     0.44
# TEXTURE                   0.31   :   COMPLEXION                0.40
# size                      0.25   :   depth                     0.22
# LIGHT                     0.23   :   TEXTURE                   0.22
#
# VAST_AMOUNT                      :   LARGE_QUANTITY                (nn_weighted_adjective_noun_identity)
# SIZE                      0.67   :   QUANTITY                  0.83
# QUANTITY                  0.64   :   SIZE                      0.70
# WIDTH                     0.45   :   WIDTH                     0.45
# COMPLEXITY                0.35   :   COMPLEXITY                0.33
# DEPTH                     0.34   :   DEPTH                     0.29
#
# EARLIER_WORK                     :   EARLY_STAGE                   (nn_weighted_adjective_noun_identity)
# importance                0.33   :   age                       0.31
# regularity                0.26   :   DURATION                  0.23
# normality                 0.24   :   crisis                    0.23
# DURATION                  0.21   :   speed                     0.22
# POSITION                  0.20   :   POSITION                  0.22
#
# GENERAL_PRINCIPLE                :   BASIC_RULE                    (nn_weighted_adjective_noun_identity)
# IMPORTANCE                0.36   :   IMPORTANCE                0.38
# NORMALITY                 0.25   :   NORMALITY                 0.35
# position                  0.23   :   COMPLEXITY                0.27
# REALITY                   0.21   :   REALITY                   0.26
# COMPLEXITY                0.16   :   consistency               0.25
#
# EFFECTIVE_WAY                    :   EFFICIENT_USE                 (nn_weighted_adjective_noun_identity)
# CONSISTENCY               0.27   :   speed                     0.38
# position                  0.25   :   CONSISTENCY               0.29
# COMPLEXITY                0.25   :   COMPLEXITY                0.28
# importance                0.25   :   REGULARITY                0.21
# REGULARITY                0.23   :   absorbency                0.20
#
# OLDER_MAN                        :   ELDERLY_WOMAN                 (nn_weighted_adjective_noun_identity)
# AGE                       0.64   :   AGE                       0.44
# complexion                0.21   :   temperature               0.17
# weight                    0.17   :   REGULARITY                0.14
# REGULARITY                0.16   :   duration                  0.14
# consistency               0.13   :   size                      0.14
#
# FEDERAL_ASSEMBLY                 :   NATIONAL_GOVERNMENT           (nn_weighted_adjective_noun_identity)
# SIZE                      0.16   :   importance                0.27
# speed                     0.15   :   SIZE                      0.23
# position                  0.14   :   crisis                    0.22
# DISTANCE                  0.13   :   quantity                  0.19
# weight                    0.13   :   DISTANCE                  0.18
#
# VAST_AMOUNT                      :   HIGH_PRICE                    (nn_weighted_adjective_noun_identity)
# SIZE                      0.67   :   QUANTITY                  0.45
# QUANTITY                  0.64   :   SIZE                      0.34
# WIDTH                     0.45   :   temperature               0.32
# complexity                0.35   :   speed                     0.30
# depth                     0.34   :   WIDTH                     0.29
#
# SOCIAL_ACTIVITY                  :   POLITICAL_ACTION              (nn_weighted_adjective_noun_identity)
# NORMALITY                 0.31   :   crisis                    0.31
# importance                0.26   :   position                  0.25
# duration                  0.26   :   reality                   0.25
# regularity                0.19   :   NORMALITY                 0.20
# complexity                0.19   :   weight                    0.20
#
# RIGHT_HAND                       :   LEFT_ARM                      (nn_weighted_adjective_noun_identity)
# POSITION                  0.84   :   POSITION                  0.57
# importance                0.28   :   distance                  0.30
# SPEED                     0.22   :   DEPTH                     0.22
# DEPTH                     0.22   :   SPEED                     0.22
# width                     0.20   :   size                      0.20
#
# VAST_AMOUNT                      :   HIGH_PRICE                    (nn_weighted_adjective_noun_identity)
# SIZE                      0.67   :   QUANTITY                  0.45
# QUANTITY                  0.64   :   SIZE                      0.34
# WIDTH                     0.45   :   temperature               0.32
# complexity                0.35   :   speed                     0.30
# depth                     0.34   :   WIDTH                     0.29
#
# OLD_PERSON                       :   ELDERLY_LADY                  (nn_weighted_adjective_noun_identity)
# AGE                       0.59   :   AGE                       0.33
# COMPLEXION                0.26   :   COMPLEXION                0.16
# duration                  0.20   :   SIZE                      0.15
# distance                  0.17   :   crisis                    0.14
# SIZE                      0.17   :   regularity                0.12
#
# GENERAL_PRINCIPLE                :   BASIC_RULE                    (nn_weighted_adjective_noun_identity)
# IMPORTANCE                0.36   :   IMPORTANCE                0.38
# NORMALITY                 0.25   :   NORMALITY                 0.35
# position                  0.23   :   COMPLEXITY                0.27
# REALITY                   0.21   :   REALITY                   0.26
# COMPLEXITY                0.16   :   consistency               0.25
#
# DIFFERENT_KIND                   :   VARIOUS_FORM                  (nn_weighted_adjective_noun_identity)
# CONSISTENCY               0.34   :   regularity                0.29
# complexity                0.33   :   CONSISTENCY               0.26
# NORMALITY                 0.33   :   NORMALITY                 0.26
# reality                   0.31   :   importance                0.23
# texture                   0.28   :   distance                  0.21
#
# EFFECTIVE_WAY                    :   EFFICIENT_USE                 (nn_weighted_adjective_noun_identity)
# CONSISTENCY               0.27   :   speed                     0.38
# position                  0.25   :   CONSISTENCY               0.29
# COMPLEXITY                0.25   :   COMPLEXITY                0.28
# importance                0.25   :   REGULARITY                0.21
# REGULARITY                0.23   :   absorbency                0.20
#
# LARGE_QUANTITY                   :   GREAT_MAJORITY                (nn_weighted_adjective_noun_identity)
# QUANTITY                  0.83   :   SIZE                      0.44
# SIZE                      0.70   :   importance                0.41
# WIDTH                     0.45   :   WIDTH                     0.28
# complexity                0.33   :   QUANTITY                  0.26
# depth                     0.29   :   weight                    0.25
#
# GOOD_PLACE                       :   HIGH_POINT                    (nn_weighted_adjective_noun_identity)
# importance                0.36   :   speed                     0.34
# CONSISTENCY               0.34   :   CONSISTENCY               0.33
# position                  0.32   :   DISTANCE                  0.32
# DISTANCE                  0.25   :   width                     0.25
# texture                   0.24   :   quantity                  0.25
#
# NEW_INFORMATION                  :   FURTHER_EVIDENCE              (nn_weighted_adjective_noun_identity)
# IMPORTANCE                0.30   :   DEPTH                     0.28
# complexity                0.28   :   quantity                  0.24
# age                       0.25   :   IMPORTANCE                0.21
# width                     0.24   :   consistency               0.19
# DEPTH                     0.24   :   weight                    0.19
#
# SOCIAL_ACTIVITY                  :   ECONOMIC_CONDITION            (nn_weighted_adjective_noun_identity)
# NORMALITY                 0.31   :   crisis                    0.36
# IMPORTANCE                0.26   :   position                  0.30
# duration                  0.26   :   IMPORTANCE                0.25
# regularity                0.19   :   NORMALITY                 0.20
# complexity                0.19   :   reality                   0.20
#
# SPECIAL_CIRCUMSTANCE             :   PARTICULAR_CASE               (nn_weighted_adjective_noun_identity)
# REGULARITY                0.22   :   IMPORTANCE                0.41
# IMPORTANCE                0.20   :   size                      0.32
# position                  0.19   :   COMPLEXITY                0.31
# COMPLEXITY                0.19   :   duration                  0.30
# age                       0.18   :   REGULARITY                0.26
#
# LARGE_NUMBER                     :   GREAT_MAJORITY                (nn_weighted_adjective_noun_identity)
# SIZE                      0.75   :   SIZE                      0.44
# QUANTITY                  0.47   :   importance                0.41
# WIDTH                     0.42   :   WIDTH                     0.28
# complexity                0.33   :   QUANTITY                  0.26
# WEIGHT                    0.31   :   WEIGHT                    0.25
#
# IMPORTANT_PART                   :   SIGNIFICANT_ROLE              (nn_weighted_adjective_noun_identity)
# IMPORTANCE                0.64   :   IMPORTANCE                0.48
# COMPLEXITY                0.29   :   position                  0.41
# duration                  0.20   :   size                      0.30
# normality                 0.19   :   COMPLEXITY                0.26
# crisis                    0.19   :   depth                     0.20
#
# OLD_PERSON                       :   ELDERLY_LADY                  (nn_weighted_adjective_noun_identity)
# AGE                       0.59   :   AGE                       0.33
# COMPLEXION                0.26   :   COMPLEXION                0.16
# duration                  0.20   :   SIZE                      0.15
# distance                  0.17   :   crisis                    0.14
# SIZE                      0.17   :   regularity                0.12
#
# EARLIER_WORK                     :   EARLY_STAGE                   (nn_weighted_adjective_noun_identity)
# importance                0.33   :   age                       0.31
# regularity                0.26   :   DURATION                  0.23
# normality                 0.24   :   crisis                    0.23
# DURATION                  0.21   :   speed                     0.22
# POSITION                  0.20   :   POSITION                  0.22
#
# GENERAL_PRINCIPLE                :   BASIC_RULE                    (nn_weighted_adjective_noun_identity)
# IMPORTANCE                0.36   :   IMPORTANCE                0.38
# NORMALITY                 0.25   :   NORMALITY                 0.35
# position                  0.23   :   COMPLEXITY                0.27
# REALITY                   0.21   :   REALITY                   0.26
# COMPLEXITY                0.16   :   consistency               0.25
#
# CERTAIN_CIRCUMSTANCE             :   PARTICULAR_CASE               (nn_weighted_adjective_noun_identity)
# REGULARITY                0.30   :   importance                0.41
# normality                 0.25   :   size                      0.32
# position                  0.25   :   complexity                0.31
# reality                   0.24   :   duration                  0.30
# crisis                    0.20   :   REGULARITY                0.26
#
# VAST_AMOUNT                      :   HIGH_PRICE                    (nn_weighted_adjective_noun_identity)
# SIZE                      0.67   :   QUANTITY                  0.45
# QUANTITY                  0.64   :   SIZE                      0.34
# WIDTH                     0.45   :   temperature               0.32
# complexity                0.35   :   speed                     0.30
# depth                     0.34   :   WIDTH                     0.29
#
# IMPORTANT_PART                   :   SIGNIFICANT_ROLE              (nn_weighted_adjective_noun_identity)
# IMPORTANCE                0.64   :   IMPORTANCE                0.48
# COMPLEXITY                0.29   :   position                  0.41
# duration                  0.20   :   size                      0.30
# normality                 0.19   :   COMPLEXITY                0.26
# crisis                    0.19   :   depth                     0.20
#
# NEW_LAW                          :   MODERN_LANGUAGE               (nn_weighted_adjective_noun_identity)
# size                      0.29   :   complexity                0.33
# AGE                       0.29   :   AGE                       0.25
# width                     0.23   :   importance                0.24
# reality                   0.19   :   SPEED                     0.23
# SPEED                     0.18   :   color                     0.22
#
# LARGE_NUMBER                     :   VAST_AMOUNT                   (nn_weighted_adjective_noun_identity)
# SIZE                      0.75   :   SIZE                      0.67
# QUANTITY                  0.47   :   QUANTITY                  0.64
# WIDTH                     0.42   :   WIDTH                     0.45
# COMPLEXITY                0.33   :   COMPLEXITY                0.35
# weight                    0.31   :   depth                     0.34
#
# OLD_PERSON                       :   ELDERLY_LADY                  (nn_weighted_adjective_noun_identity)
# AGE                       0.59   :   AGE                       0.33
# COMPLEXION                0.26   :   COMPLEXION                0.16
# duration                  0.20   :   SIZE                      0.15
# distance                  0.17   :   crisis                    0.14
# SIZE                      0.17   :   regularity                0.12
#
# NEW_LIFE                         :   EARLY_AGE                     (nn_weighted_adjective_noun_identity)
# DURATION                  0.39   :   AGE                       0.81
# AGE                       0.31   :   DURATION                  0.25
# size                      0.29   :   temperature               0.25
# reality                   0.28   :   speed                     0.22
# complexity                0.24   :   distance                  0.21
#
# POLITICAL_ACTION                 :   ECONOMIC_DEVELOPMENT          (nn_weighted_adjective_noun_identity)
# CRISIS                    0.31   :   CRISIS                    0.34
# position                  0.25   :   importance                0.20
# REALITY                   0.25   :   size                      0.19
# normality                 0.20   :   speed                     0.16
# weight                    0.20   :   REALITY                   0.16
#
# LARGE_NUMBER                     :   GREAT_MAJORITY                (nn_weighted_adjective_noun_identity)
# SIZE                      0.75   :   SIZE                      0.44
# QUANTITY                  0.47   :   importance                0.41
# WIDTH                     0.42   :   WIDTH                     0.28
# complexity                0.33   :   QUANTITY                  0.26
# WEIGHT                    0.31   :   WEIGHT                    0.25
#
# CERTAIN_CIRCUMSTANCE             :   PARTICULAR_CASE               (nn_weighted_adjective_noun_identity)
# REGULARITY                0.30   :   importance                0.41
# normality                 0.25   :   size                      0.32
# position                  0.25   :   complexity                0.31
# reality                   0.24   :   duration                  0.30
# crisis                    0.20   :   REGULARITY                0.26
#
# CERTAIN_CIRCUMSTANCE             :   PARTICULAR_CASE               (nn_weighted_adjective_noun_identity)
# REGULARITY                0.30   :   importance                0.41
# normality                 0.25   :   size                      0.32
# position                  0.25   :   complexity                0.31
# reality                   0.24   :   duration                  0.30
# crisis                    0.20   :   REGULARITY                0.26
#
# IMPORTANT_PART                   :   SIGNIFICANT_ROLE              (nn_weighted_adjective_noun_identity)
# IMPORTANCE                0.64   :   IMPORTANCE                0.48
# COMPLEXITY                0.29   :   position                  0.41
# duration                  0.20   :   size                      0.30
# normality                 0.19   :   COMPLEXITY                0.26
# crisis                    0.19   :   depth                     0.20
#
# DIFFERENT_KIND                   :   VARIOUS_FORM                  (nn_weighted_adjective_noun_identity)
# CONSISTENCY               0.34   :   regularity                0.29
# complexity                0.33   :   CONSISTENCY               0.26
# NORMALITY                 0.33   :   NORMALITY                 0.26
# reality                   0.31   :   importance                0.23
# texture                   0.28   :   distance                  0.21
#
# EFFECTIVE_WAY                    :   EFFICIENT_USE                 (nn_weighted_adjective_noun_identity)
# CONSISTENCY               0.27   :   speed                     0.38
# position                  0.25   :   CONSISTENCY               0.29
# COMPLEXITY                0.25   :   COMPLEXITY                0.28
# importance                0.25   :   REGULARITY                0.21
# REGULARITY                0.23   :   absorbency                0.20
#
# OLDER_MAN                        :   ELDERLY_WOMAN                 (nn_weighted_adjective_noun_identity)
# AGE                       0.64   :   AGE                       0.44
# complexion                0.21   :   temperature               0.17
# weight                    0.17   :   REGULARITY                0.14
# REGULARITY                0.16   :   duration                  0.14
# consistency               0.13   :   size                      0.14
#
# FEDERAL_ASSEMBLY                 :   NATIONAL_GOVERNMENT           (nn_weighted_adjective_noun_identity)
# SIZE                      0.16   :   importance                0.27
# speed                     0.15   :   SIZE                      0.23
# position                  0.14   :   crisis                    0.22
# DISTANCE                  0.13   :   quantity                  0.19
# weight                    0.13   :   DISTANCE                  0.18
#
# LARGE_QUANTITY                   :   GREAT_MAJORITY                (nn_weighted_adjective_noun_identity)
# QUANTITY                  0.83   :   SIZE                      0.44
# SIZE                      0.70   :   importance                0.41
# WIDTH                     0.45   :   WIDTH                     0.28
# complexity                0.33   :   QUANTITY                  0.26
# depth                     0.29   :   weight                    0.25
#
# VAST_AMOUNT                      :   LARGE_QUANTITY                (nn_weighted_adjective_noun_identity)
# SIZE                      0.67   :   QUANTITY                  0.83
# QUANTITY                  0.64   :   SIZE                      0.70
# WIDTH                     0.45   :   WIDTH                     0.45
# COMPLEXITY                0.35   :   COMPLEXITY                0.33
# DEPTH                     0.34   :   DEPTH                     0.29
#
# SPECIAL_CIRCUMSTANCE             :   PARTICULAR_CASE               (nn_weighted_adjective_noun_identity)
# REGULARITY                0.22   :   IMPORTANCE                0.41
# IMPORTANCE                0.20   :   size                      0.32
# position                  0.19   :   COMPLEXITY                0.31
# COMPLEXITY                0.19   :   duration                  0.30
# age                       0.18   :   REGULARITY                0.26
#
# LOCAL_OFFICE                     :   NEW_TECHNOLOGY                (nn_weighted_adjective_noun_identity)
# SIZE                      0.30   :   complexity                0.35
# distance                  0.20   :   speed                     0.32
# position                  0.20   :   SIZE                      0.31
# importance                0.18   :   width                     0.26
# temperature               0.18   :   age                       0.25
#
# CERTAIN_CIRCUMSTANCE             :   PARTICULAR_CASE               (nn_weighted_adjective_noun_identity)
# REGULARITY                0.30   :   importance                0.41
# normality                 0.25   :   size                      0.32
# position                  0.25   :   complexity                0.31
# reality                   0.24   :   duration                  0.30
# crisis                    0.20   :   REGULARITY                0.26
#
# VAST_AMOUNT                      :   HIGH_PRICE                    (nn_weighted_adjective_noun_identity)
# SIZE                      0.67   :   QUANTITY                  0.45
# QUANTITY                  0.64   :   SIZE                      0.34
# WIDTH                     0.45   :   temperature               0.32
# complexity                0.35   :   speed                     0.30
# depth                     0.34   :   WIDTH                     0.29
#
# IMPORTANT_PART                   :   SIGNIFICANT_ROLE              (nn_weighted_adjective_noun_identity)
# IMPORTANCE                0.64   :   IMPORTANCE                0.48
# COMPLEXITY                0.29   :   position                  0.41
# duration                  0.20   :   size                      0.30
# normality                 0.19   :   COMPLEXITY                0.26
# crisis                    0.19   :   depth                     0.20
#
# EARLIER_WORK                     :   EARLY_STAGE                   (nn_weighted_adjective_noun_identity)
# importance                0.33   :   age                       0.31
# regularity                0.26   :   DURATION                  0.23
# normality                 0.24   :   crisis                    0.23
# DURATION                  0.21   :   speed                     0.22
# POSITION                  0.20   :   POSITION                  0.22
#
# GENERAL_PRINCIPLE                :   BASIC_RULE                    (nn_weighted_adjective_noun_identity)
# IMPORTANCE                0.36   :   IMPORTANCE                0.38
# NORMALITY                 0.25   :   NORMALITY                 0.35
# position                  0.23   :   COMPLEXITY                0.27
# REALITY                   0.21   :   REALITY                   0.26
# COMPLEXITY                0.16   :   consistency               0.25
#
# LARGE_QUANTITY                   :   GREAT_MAJORITY                (nn_weighted_adjective_noun_identity)
# QUANTITY                  0.83   :   SIZE                      0.44
# SIZE                      0.70   :   importance                0.41
# WIDTH                     0.45   :   WIDTH                     0.28
# complexity                0.33   :   QUANTITY                  0.26
# depth                     0.29   :   weight                    0.25
#
# EFFECTIVE_WAY                    :   EFFICIENT_USE                 (nn_weighted_adjective_noun_identity)
# CONSISTENCY               0.27   :   speed                     0.38
# position                  0.25   :   CONSISTENCY               0.29
# COMPLEXITY                0.25   :   COMPLEXITY                0.28
# importance                0.25   :   REGULARITY                0.21
# REGULARITY                0.23   :   absorbency                0.20
#
# OLDER_MAN                        :   ELDERLY_WOMAN                 (nn_weighted_adjective_noun_identity)
# AGE                       0.64   :   AGE                       0.44
# complexion                0.21   :   temperature               0.17
# weight                    0.17   :   REGULARITY                0.14
# REGULARITY                0.16   :   duration                  0.14
# consistency               0.13   :   size                      0.14
#
# FEDERAL_ASSEMBLY                 :   NATIONAL_GOVERNMENT           (nn_weighted_adjective_noun_identity)
# SIZE                      0.16   :   importance                0.27
# speed                     0.15   :   SIZE                      0.23
# position                  0.14   :   crisis                    0.22
# DISTANCE                  0.13   :   quantity                  0.19
# weight                    0.13   :   DISTANCE                  0.18
#
# LARGE_NUMBER                     :   VAST_AMOUNT                   (nn_weighted_adjective_noun_identity)
# SIZE                      0.75   :   SIZE                      0.67
# QUANTITY                  0.47   :   QUANTITY                  0.64
# WIDTH                     0.42   :   WIDTH                     0.45
# COMPLEXITY                0.33   :   COMPLEXITY                0.35
# weight                    0.31   :   depth                     0.34
#
# OLDER_MAN                        :   ELDERLY_WOMAN                 (nn_weighted_adjective_noun_identity)
# AGE                       0.64   :   AGE                       0.44
# complexion                0.21   :   temperature               0.17
# weight                    0.17   :   REGULARITY                0.14
# REGULARITY                0.16   :   duration                  0.14
# consistency               0.13   :   size                      0.14
#
# FEDERAL_ASSEMBLY                 :   NATIONAL_GOVERNMENT           (nn_weighted_adjective_noun_identity)
# SIZE                      0.16   :   importance                0.27
# speed                     0.15   :   SIZE                      0.23
# position                  0.14   :   crisis                    0.22
# DISTANCE                  0.13   :   quantity                  0.19
# weight                    0.13   :   DISTANCE                  0.18
#
# LARGE_NUMBER                     :   GREAT_MAJORITY                (nn_weighted_adjective_noun_identity)
# SIZE                      0.75   :   SIZE                      0.44
# QUANTITY                  0.47   :   importance                0.41
# WIDTH                     0.42   :   WIDTH                     0.28
# complexity                0.33   :   QUANTITY                  0.26
# WEIGHT                    0.31   :   WEIGHT                    0.25
#
# CERTAIN_CIRCUMSTANCE             :   PARTICULAR_CASE               (nn_weighted_adjective_noun_identity)
# REGULARITY                0.30   :   importance                0.41
# normality                 0.25   :   size                      0.32
# position                  0.25   :   complexity                0.31
# reality                   0.24   :   duration                  0.30
# crisis                    0.20   :   REGULARITY                0.26
#
# IMPORTANT_PART                   :   SIGNIFICANT_ROLE              (nn_weighted_adjective_noun_identity)
# IMPORTANCE                0.64   :   IMPORTANCE                0.48
# COMPLEXITY                0.29   :   position                  0.41
# duration                  0.20   :   size                      0.30
# normality                 0.19   :   COMPLEXITY                0.26
# crisis                    0.19   :   depth                     0.20
#
# LARGE_NUMBER                     :   VAST_AMOUNT                   (nn_weighted_adjective_noun_identity)
# SIZE                      0.75   :   SIZE                      0.67
# QUANTITY                  0.47   :   QUANTITY                  0.64
# WIDTH                     0.42   :   WIDTH                     0.45
# COMPLEXITY                0.33   :   COMPLEXITY                0.35
# weight                    0.31   :   depth                     0.34
#
# DIFFERENT_KIND                   :   VARIOUS_FORM                  (nn_weighted_adjective_noun_identity)
# CONSISTENCY               0.34   :   regularity                0.29
# complexity                0.33   :   CONSISTENCY               0.26
# NORMALITY                 0.33   :   NORMALITY                 0.26
# reality                   0.31   :   importance                0.23
# texture                   0.28   :   distance                  0.21
#
# OLDER_MAN                        :   ELDERLY_WOMAN                 (nn_weighted_adjective_noun_identity)
# AGE                       0.64   :   AGE                       0.44
# complexion                0.21   :   temperature               0.17
# weight                    0.17   :   REGULARITY                0.14
# REGULARITY                0.16   :   duration                  0.14
# consistency               0.13   :   size                      0.14
#
# IMPORTANT_PART                   :   SIGNIFICANT_ROLE              (nn_weighted_adjective_noun_identity)
# IMPORTANCE                0.64   :   IMPORTANCE                0.48
# COMPLEXITY                0.29   :   position                  0.41
# duration                  0.20   :   size                      0.30
# normality                 0.19   :   COMPLEXITY                0.26
# crisis                    0.19   :   depth                     0.20
#
# OLD_PERSON                       :   ELDERLY_LADY                  (nn_weighted_adjective_noun_identity)
# AGE                       0.59   :   AGE                       0.33
# COMPLEXION                0.26   :   COMPLEXION                0.16
# duration                  0.20   :   SIZE                      0.15
# distance                  0.17   :   crisis                    0.14
# SIZE                      0.17   :   regularity                0.12
#
# EFFECTIVE_WAY                    :   EFFICIENT_USE                 (nn_weighted_adjective_noun_identity)
# CONSISTENCY               0.27   :   speed                     0.38
# position                  0.25   :   CONSISTENCY               0.29
# COMPLEXITY                0.25   :   COMPLEXITY                0.28
# importance                0.25   :   REGULARITY                0.21
# REGULARITY                0.23   :   absorbency                0.20
#
# LARGE_NUMBER                     :   GREAT_MAJORITY                (nn_weighted_adjective_noun_identity)
# SIZE                      0.75   :   SIZE                      0.44
# QUANTITY                  0.47   :   importance                0.41
# WIDTH                     0.42   :   WIDTH                     0.28
# complexity                0.33   :   QUANTITY                  0.26
# WEIGHT                    0.31   :   WEIGHT                    0.25
#
# CERTAIN_CIRCUMSTANCE             :   PARTICULAR_CASE               (nn_weighted_adjective_noun_identity)
# REGULARITY                0.30   :   importance                0.41
# normality                 0.25   :   size                      0.32
# position                  0.25   :   complexity                0.31
# reality                   0.24   :   duration                  0.30
# crisis                    0.20   :   REGULARITY                0.26
#
# NEW_INFORMATION                  :   FURTHER_EVIDENCE              (nn_weighted_adjective_noun_identity)
# IMPORTANCE                0.30   :   DEPTH                     0.28
# complexity                0.28   :   quantity                  0.24
# age                       0.25   :   IMPORTANCE                0.21
# width                     0.24   :   consistency               0.19
# DEPTH                     0.24   :   weight                    0.19
#
# VAST_AMOUNT                      :   LARGE_QUANTITY                (nn_weighted_adjective_noun_identity)
# SIZE                      0.67   :   QUANTITY                  0.83
# QUANTITY                  0.64   :   SIZE                      0.70
# WIDTH                     0.45   :   WIDTH                     0.45
# COMPLEXITY                0.35   :   COMPLEXITY                0.33
# DEPTH                     0.34   :   DEPTH                     0.29
#
# SPECIAL_CIRCUMSTANCE             :   PARTICULAR_CASE               (nn_weighted_adjective_noun_identity)
# REGULARITY                0.22   :   IMPORTANCE                0.41
# IMPORTANCE                0.20   :   size                      0.32
# position                  0.19   :   COMPLEXITY                0.31
# COMPLEXITY                0.19   :   duration                  0.30
# age                       0.18   :   REGULARITY                0.26
#
# GOOD_PLACE                       :   HIGH_POINT                    (nn_weighted_adjective_noun_identity)
# importance                0.36   :   speed                     0.34
# CONSISTENCY               0.34   :   CONSISTENCY               0.33
# position                  0.32   :   DISTANCE                  0.32
# DISTANCE                  0.25   :   width                     0.25
# texture                   0.24   :   quantity                  0.25
#
# CERTAIN_CIRCUMSTANCE             :   PARTICULAR_CASE               (nn_weighted_adjective_noun_identity)
# REGULARITY                0.30   :   importance                0.41
# normality                 0.25   :   size                      0.32
# position                  0.25   :   complexity                0.31
# reality                   0.24   :   duration                  0.30
# crisis                    0.20   :   REGULARITY                0.26
#
# VAST_AMOUNT                      :   HIGH_PRICE                    (nn_weighted_adjective_noun_identity)
# SIZE                      0.67   :   QUANTITY                  0.45
# QUANTITY                  0.64   :   SIZE                      0.34
# WIDTH                     0.45   :   temperature               0.32
# complexity                0.35   :   speed                     0.30
# depth                     0.34   :   WIDTH                     0.29
#
# NEW_LIFE                         :   ECONOMIC_DEVELOPMENT          (nn_weighted_adjective_noun_identity)
# duration                  0.39   :   crisis                    0.34
# age                       0.31   :   importance                0.20
# SIZE                      0.29   :   SIZE                      0.19
# REALITY                   0.28   :   speed                     0.16
# complexity                0.24   :   REALITY                   0.16
#
# LARGE_NUMBER                     :   VAST_AMOUNT                   (nn_weighted_adjective_noun_identity)
# SIZE                      0.75   :   SIZE                      0.67
# QUANTITY                  0.47   :   QUANTITY                  0.64
# WIDTH                     0.42   :   WIDTH                     0.45
# COMPLEXITY                0.33   :   COMPLEXITY                0.35
# weight                    0.31   :   depth                     0.34
#
# DIFFERENT_KIND                   :   VARIOUS_FORM                  (nn_weighted_adjective_noun_identity)
# CONSISTENCY               0.34   :   regularity                0.29
# complexity                0.33   :   CONSISTENCY               0.26
# NORMALITY                 0.33   :   NORMALITY                 0.26
# reality                   0.31   :   importance                0.23
# texture                   0.28   :   distance                  0.21
#
# FEDERAL_ASSEMBLY                 :   NATIONAL_GOVERNMENT           (nn_weighted_adjective_noun_identity)
# SIZE                      0.16   :   importance                0.27
# speed                     0.15   :   SIZE                      0.23
# position                  0.14   :   crisis                    0.22
# DISTANCE                  0.13   :   quantity                  0.19
# weight                    0.13   :   DISTANCE                  0.18
# """
#
# import re
#
# regex = "(\w+\s{5,}:\s+\w+\s+\(\w+\))"
# # regex = re.compile(regex)
# # result = regex.match(complete_human_ratings_string)
# # print()
#
#
# # print(re.match(regex, complete_human_ratings_string))
# #


#
# complete_human_ratings_string_cutoff4 = """
# VAST_AMOUNT                      :   LARGE_QUANTITY                (nn_weighted_adjective_noun_identity)
# QUANTITY                  0.58   :   QUANTITY                  0.78
# SIZE                      0.57   :   SIZE                      0.63
# WIDTH                     0.42   :   WIDTH                     0.40
# THICKNESS                 0.38   :   THICKNESS                 0.38
# substantiality            0.35   :   volume                    0.36
#
# VAST_AMOUNT                      :   LARGE_QUANTITY                (nn_weighted_adjective_noun_identity)
# QUANTITY                  0.58   :   QUANTITY                  0.78
# SIZE                      0.57   :   SIZE                      0.63
# WIDTH                     0.42   :   WIDTH                     0.40
# THICKNESS                 0.38   :   THICKNESS                 0.38
# substantiality            0.35   :   volume                    0.36
#
# IMPORTANT_PART                   :   SIGNIFICANT_ROLE              (nn_weighted_adjective_noun_identity)
# IMPORTANCE                0.62   :   SIGNIFICANCE              0.56
# SIGNIFICANCE              0.54   :   IMPORTANCE                0.53
# necessity                 0.41   :   status                    0.40
# pride                     0.39   :   RESPONSIBILITY            0.36
# RESPONSIBILITY            0.35   :   likelihood                0.36
#
# LARGE_NUMBER                     :   VAST_AMOUNT                   (nn_weighted_adjective_noun_identity)
# SIZE                      0.67   :   QUANTITY                  0.58
# QUANTITY                  0.45   :   SIZE                      0.57
# WIDTH                     0.38   :   WIDTH                     0.42
# volume                    0.38   :   thickness                 0.38
# complexity                0.34   :   substantiality            0.35
#
# GENERAL_PRINCIPLE                :   BASIC_RULE                    (nn_weighted_adjective_noun_identity)
# GENERALITY                0.58   :   GENERALITY                0.44
# concreteness              0.35   :   humaneness                0.43
# equality                  0.33   :   practicality              0.43
# humanness                 0.33   :   comprehensiveness         0.41
# morality                  0.33   :   reasonableness            0.41
#
# EFFECTIVE_WAY                    :   EFFICIENT_USE                 (nn_weighted_adjective_noun_identity)
# EFFECTIVENESS             0.75   :   EFFECTIVENESS             0.48
# efficacy                  0.48   :   PRACTICALITY              0.39
# potency                   0.44   :   ACCURACY                  0.38
# PRACTICALITY              0.43   :   quality                   0.34
# ACCURACY                  0.41   :   utility                   0.34
#
# FEDERAL_ASSEMBLY                 :   NATIONAL_GOVERNMENT           (nn_weighted_adjective_noun_identity)
# majority                  0.23   :   importance                0.35
# otherness                 0.22   :   significance              0.31
# nature                    0.22   :   stature                   0.30
# seniority                 0.21   :   seriousness               0.29
# sameness                  0.21   :   intrusiveness             0.29
#
# CERTAIN_CIRCUMSTANCE             :   PARTICULAR_CASE               (nn_weighted_adjective_noun_identity)
# COMMONNESS                0.42   :   significance              0.49
# otherness                 0.40   :   importance                0.45
# foreignness               0.39   :   COMMONNESS                0.43
# concreteness              0.37   :   abstractness              0.41
# ordinariness              0.37   :   seriousness               0.40
#
# VAST_AMOUNT                      :   HIGH_PRICE                    (nn_weighted_adjective_noun_identity)
# QUANTITY                  0.58   :   degree                    0.54
# size                      0.57   :   height                    0.48
# width                     0.42   :   price                     0.46
# thickness                 0.38   :   moderation                0.44
# substantiality            0.35   :   QUANTITY                  0.33
#
# IMPORTANT_PART                   :   SIGNIFICANT_ROLE              (nn_weighted_adjective_noun_identity)
# IMPORTANCE                0.62   :   SIGNIFICANCE              0.56
# SIGNIFICANCE              0.54   :   IMPORTANCE                0.53
# necessity                 0.41   :   status                    0.40
# pride                     0.39   :   RESPONSIBILITY            0.36
# RESPONSIBILITY            0.35   :   likelihood                0.36
#
# LARGE_NUMBER                     :   VAST_AMOUNT                   (nn_weighted_adjective_noun_identity)
# SIZE                      0.67   :   QUANTITY                  0.58
# QUANTITY                  0.45   :   SIZE                      0.57
# WIDTH                     0.38   :   WIDTH                     0.42
# volume                    0.38   :   thickness                 0.38
# complexity                0.34   :   substantiality            0.35
#
# FEDERAL_ASSEMBLY                 :   NATIONAL_GOVERNMENT           (nn_weighted_adjective_noun_identity)
# majority                  0.23   :   importance                0.35
# otherness                 0.22   :   significance              0.31
# nature                    0.22   :   stature                   0.30
# seniority                 0.21   :   seriousness               0.29
# sameness                  0.21   :   intrusiveness             0.29
#
# IMPORTANT_PART                   :   SIGNIFICANT_ROLE              (nn_weighted_adjective_noun_identity)
# IMPORTANCE                0.62   :   SIGNIFICANCE              0.56
# SIGNIFICANCE              0.54   :   IMPORTANCE                0.53
# necessity                 0.41   :   status                    0.40
# pride                     0.39   :   RESPONSIBILITY            0.36
# RESPONSIBILITY            0.35   :   likelihood                0.36
#
# CERTAIN_CIRCUMSTANCE             :   PARTICULAR_CASE               (nn_weighted_adjective_noun_identity)
# COMMONNESS                0.42   :   significance              0.49
# otherness                 0.40   :   importance                0.45
# foreignness               0.39   :   COMMONNESS                0.43
# concreteness              0.37   :   abstractness              0.41
# ordinariness              0.37   :   seriousness               0.40
#
# IMPORTANT_PART                   :   SIGNIFICANT_ROLE              (nn_weighted_adjective_noun_identity)
# IMPORTANCE                0.62   :   SIGNIFICANCE              0.56
# SIGNIFICANCE              0.54   :   IMPORTANCE                0.53
# necessity                 0.41   :   status                    0.40
# pride                     0.39   :   RESPONSIBILITY            0.36
# RESPONSIBILITY            0.35   :   likelihood                0.36
#
# LARGE_NUMBER                     :   VAST_AMOUNT                   (nn_weighted_adjective_noun_identity)
# SIZE                      0.67   :   QUANTITY                  0.58
# QUANTITY                  0.45   :   SIZE                      0.57
# WIDTH                     0.38   :   WIDTH                     0.42
# volume                    0.38   :   thickness                 0.38
# complexity                0.34   :   substantiality            0.35
#
# GENERAL_PRINCIPLE                :   BASIC_RULE                    (nn_weighted_adjective_noun_identity)
# GENERALITY                0.58   :   GENERALITY                0.44
# concreteness              0.35   :   humaneness                0.43
# equality                  0.33   :   practicality              0.43
# humanness                 0.33   :   comprehensiveness         0.41
# morality                  0.33   :   reasonableness            0.41
#
# DIFFERENT_KIND                   :   VARIOUS_FORM                  (nn_weighted_adjective_noun_identity)
# SAMENESS                  0.57   :   OTHERNESS                 0.39
# similarity                0.54   :   COMMONALITY               0.34
# COMMONALITY               0.50   :   foreignness               0.33
# DIFFERENCE                0.50   :   SAMENESS                  0.33
# OTHERNESS                 0.48   :   DIFFERENCE                0.33
#
# VAST_AMOUNT                      :   LARGE_QUANTITY                (nn_weighted_adjective_noun_identity)
# QUANTITY                  0.58   :   QUANTITY                  0.78
# SIZE                      0.57   :   SIZE                      0.63
# WIDTH                     0.42   :   WIDTH                     0.40
# THICKNESS                 0.38   :   THICKNESS                 0.38
# substantiality            0.35   :   volume                    0.36
#
# VAST_AMOUNT                      :   LARGE_QUANTITY                (nn_weighted_adjective_noun_identity)
# QUANTITY                  0.58   :   QUANTITY                  0.78
# SIZE                      0.57   :   SIZE                      0.63
# WIDTH                     0.42   :   WIDTH                     0.40
# THICKNESS                 0.38   :   THICKNESS                 0.38
# substantiality            0.35   :   volume                    0.36
#
# SOCIAL_EVENT                     :   SPECIAL_CIRCUMSTANCE          (nn_weighted_adjective_noun_identity)
# sociality                 0.46   :   ordinariness              0.47
# sociability               0.42   :   commonness                0.44
# equality                  0.35   :   OTHERNESS                 0.40
# morality                  0.31   :   nobility                  0.37
# OTHERNESS                 0.30   :   gregariousness            0.37
#
# LARGE_NUMBER                     :   VAST_AMOUNT                   (nn_weighted_adjective_noun_identity)
# SIZE                      0.67   :   QUANTITY                  0.58
# QUANTITY                  0.45   :   SIZE                      0.57
# WIDTH                     0.38   :   WIDTH                     0.42
# volume                    0.38   :   thickness                 0.38
# complexity                0.34   :   substantiality            0.35
#
# GENERAL_PRINCIPLE                :   BASIC_RULE                    (nn_weighted_adjective_noun_identity)
# GENERALITY                0.58   :   GENERALITY                0.44
# concreteness              0.35   :   humaneness                0.43
# equality                  0.33   :   practicality              0.43
# humanness                 0.33   :   comprehensiveness         0.41
# morality                  0.33   :   reasonableness            0.41
#
# FEDERAL_ASSEMBLY                 :   NATIONAL_GOVERNMENT           (nn_weighted_adjective_noun_identity)
# majority                  0.23   :   importance                0.35
# otherness                 0.22   :   significance              0.31
# nature                    0.22   :   stature                   0.30
# seniority                 0.21   :   seriousness               0.29
# sameness                  0.21   :   intrusiveness             0.29
#
# VAST_AMOUNT                      :   LARGE_QUANTITY                (nn_weighted_adjective_noun_identity)
# QUANTITY                  0.58   :   QUANTITY                  0.78
# SIZE                      0.57   :   SIZE                      0.63
# WIDTH                     0.42   :   WIDTH                     0.40
# THICKNESS                 0.38   :   THICKNESS                 0.38
# substantiality            0.35   :   volume                    0.36
#
# SPECIAL_CIRCUMSTANCE             :   PARTICULAR_CASE               (nn_weighted_adjective_noun_identity)
# ordinariness              0.47   :   significance              0.49
# COMMONNESS                0.44   :   importance                0.45
# otherness                 0.40   :   COMMONNESS                0.43
# nobility                  0.37   :   abstractness              0.41
# gregariousness            0.37   :   seriousness               0.40
#
# VAST_AMOUNT                      :   LARGE_QUANTITY                (nn_weighted_adjective_noun_identity)
# QUANTITY                  0.58   :   QUANTITY                  0.78
# SIZE                      0.57   :   SIZE                      0.63
# WIDTH                     0.42   :   WIDTH                     0.40
# THICKNESS                 0.38   :   THICKNESS                 0.38
# substantiality            0.35   :   volume                    0.36
#
# LARGE_NUMBER                     :   VAST_AMOUNT                   (nn_weighted_adjective_noun_identity)
# SIZE                      0.67   :   QUANTITY                  0.58
# QUANTITY                  0.45   :   SIZE                      0.57
# WIDTH                     0.38   :   WIDTH                     0.42
# volume                    0.38   :   thickness                 0.38
# complexity                0.34   :   substantiality            0.35
#
# OLD_PERSON                       :   ELDERLY_LADY                  (nn_weighted_adjective_noun_identity)
# AGE                       0.45   :   commonness                0.31
# otherness                 0.40   :   AGE                       0.30
# typicality                0.40   :   health                    0.29
# abstractness              0.37   :   thoughtfulness            0.29
# originality               0.36   :   kindness                  0.29
#
# DIFFERENT_KIND                   :   VARIOUS_FORM                  (nn_weighted_adjective_noun_identity)
# SAMENESS                  0.57   :   OTHERNESS                 0.39
# similarity                0.54   :   COMMONALITY               0.34
# COMMONALITY               0.50   :   foreignness               0.33
# DIFFERENCE                0.50   :   SAMENESS                  0.33
# OTHERNESS                 0.48   :   DIFFERENCE                0.33
#
# LARGE_QUANTITY                   :   GREAT_MAJORITY                (nn_weighted_adjective_noun_identity)
# quantity                  0.78   :   majority                  0.70
# SIZE                      0.63   :   quality                   0.41
# width                     0.40   :   minority                  0.41
# thickness                 0.38   :   SIZE                      0.32
# volume                    0.36   :   cleanness                 0.32
#
# MAJOR_ISSUE                      :   AMERICAN_COUNTRY              (nn_weighted_adjective_noun_identity)
# significance              0.46   :   FOREIGNNESS               0.32
# importance                0.44   :   corruptness               0.27
# size                      0.35   :   cleanness                 0.25
# stature                   0.34   :   purity                    0.25
# FOREIGNNESS               0.32   :   niceness                  0.25
#
# LARGE_NUMBER                     :   GREAT_MAJORITY                (nn_weighted_adjective_noun_identity)
# SIZE                      0.67   :   majority                  0.70
# quantity                  0.45   :   quality                   0.41
# width                     0.38   :   minority                  0.41
# volume                    0.38   :   SIZE                      0.32
# complexity                0.34   :   cleanness                 0.32
#
# CERTAIN_CIRCUMSTANCE             :   PARTICULAR_CASE               (nn_weighted_adjective_noun_identity)
# COMMONNESS                0.42   :   significance              0.49
# otherness                 0.40   :   importance                0.45
# foreignness               0.39   :   COMMONNESS                0.43
# concreteness              0.37   :   abstractness              0.41
# ordinariness              0.37   :   seriousness               0.40
#
# IMPORTANT_PART                   :   SIGNIFICANT_ROLE              (nn_weighted_adjective_noun_identity)
# IMPORTANCE                0.62   :   SIGNIFICANCE              0.56
# SIGNIFICANCE              0.54   :   IMPORTANCE                0.53
# necessity                 0.41   :   status                    0.40
# pride                     0.39   :   RESPONSIBILITY            0.36
# RESPONSIBILITY            0.35   :   likelihood                0.36
#
# LARGE_NUMBER                     :   GREAT_MAJORITY                (nn_weighted_adjective_noun_identity)
# SIZE                      0.67   :   majority                  0.70
# quantity                  0.45   :   quality                   0.41
# width                     0.38   :   minority                  0.41
# volume                    0.38   :   SIZE                      0.32
# complexity                0.34   :   cleanness                 0.32
#
# CERTAIN_CIRCUMSTANCE             :   PARTICULAR_CASE               (nn_weighted_adjective_noun_identity)
# COMMONNESS                0.42   :   significance              0.49
# otherness                 0.40   :   importance                0.45
# foreignness               0.39   :   COMMONNESS                0.43
# concreteness              0.37   :   abstractness              0.41
# ordinariness              0.37   :   seriousness               0.40
#
# IMPORTANT_PART                   :   SIGNIFICANT_ROLE              (nn_weighted_adjective_noun_identity)
# IMPORTANCE                0.62   :   SIGNIFICANCE              0.56
# SIGNIFICANCE              0.54   :   IMPORTANCE                0.53
# necessity                 0.41   :   status                    0.40
# pride                     0.39   :   RESPONSIBILITY            0.36
# RESPONSIBILITY            0.35   :   likelihood                0.36
#
# IMPORTANT_PART                   :   SIGNIFICANT_ROLE              (nn_weighted_adjective_noun_identity)
# IMPORTANCE                0.62   :   SIGNIFICANCE              0.56
# SIGNIFICANCE              0.54   :   IMPORTANCE                0.53
# necessity                 0.41   :   status                    0.40
# pride                     0.39   :   RESPONSIBILITY            0.36
# RESPONSIBILITY            0.35   :   likelihood                0.36
#
# LARGE_NUMBER                     :   VAST_AMOUNT                   (nn_weighted_adjective_noun_identity)
# SIZE                      0.67   :   QUANTITY                  0.58
# QUANTITY                  0.45   :   SIZE                      0.57
# WIDTH                     0.38   :   WIDTH                     0.42
# volume                    0.38   :   thickness                 0.38
# complexity                0.34   :   substantiality            0.35
#
# LARGE_NUMBER                     :   VAST_AMOUNT                   (nn_weighted_adjective_noun_identity)
# SIZE                      0.67   :   QUANTITY                  0.58
# QUANTITY                  0.45   :   SIZE                      0.57
# WIDTH                     0.38   :   WIDTH                     0.42
# volume                    0.38   :   thickness                 0.38
# complexity                0.34   :   substantiality            0.35
#
# VAST_AMOUNT                      :   LARGE_QUANTITY                (nn_weighted_adjective_noun_identity)
# QUANTITY                  0.58   :   QUANTITY                  0.78
# SIZE                      0.57   :   SIZE                      0.63
# WIDTH                     0.42   :   WIDTH                     0.40
# THICKNESS                 0.38   :   THICKNESS                 0.38
# substantiality            0.35   :   volume                    0.36
#
# VAST_AMOUNT                      :   LARGE_QUANTITY                (nn_weighted_adjective_noun_identity)
# QUANTITY                  0.58   :   QUANTITY                  0.78
# SIZE                      0.57   :   SIZE                      0.63
# WIDTH                     0.42   :   WIDTH                     0.40
# THICKNESS                 0.38   :   THICKNESS                 0.38
# substantiality            0.35   :   volume                    0.36
#
# NEW_INFORMATION                  :   FURTHER_EVIDENCE              (nn_weighted_adjective_noun_identity)
# individuality             0.46   :   possibility               0.36
# originality               0.44   :   finality                  0.36
# permanence                0.44   :   credibility               0.36
# immediacy                 0.44   :   sharpness                 0.34
# correctness               0.42   :   freshness                 0.34
#
# VAST_AMOUNT                      :   LARGE_QUANTITY                (nn_weighted_adjective_noun_identity)
# QUANTITY                  0.58   :   QUANTITY                  0.78
# SIZE                      0.57   :   SIZE                      0.63
# WIDTH                     0.42   :   WIDTH                     0.40
# THICKNESS                 0.38   :   THICKNESS                 0.38
# substantiality            0.35   :   volume                    0.36
#
# LARGE_NUMBER                     :   VAST_AMOUNT                   (nn_weighted_adjective_noun_identity)
# SIZE                      0.67   :   QUANTITY                  0.58
# QUANTITY                  0.45   :   SIZE                      0.57
# WIDTH                     0.38   :   WIDTH                     0.42
# volume                    0.38   :   thickness                 0.38
# complexity                0.34   :   substantiality            0.35
#
# LARGE_QUANTITY                   :   GREAT_MAJORITY                (nn_weighted_adjective_noun_identity)
# quantity                  0.78   :   majority                  0.70
# SIZE                      0.63   :   quality                   0.41
# width                     0.40   :   minority                  0.41
# thickness                 0.38   :   SIZE                      0.32
# volume                    0.36   :   cleanness                 0.32
#
# LARGE_NUMBER                     :   GREAT_MAJORITY                (nn_weighted_adjective_noun_identity)
# SIZE                      0.67   :   majority                  0.70
# quantity                  0.45   :   quality                   0.41
# width                     0.38   :   minority                  0.41
# volume                    0.38   :   SIZE                      0.32
# complexity                0.34   :   cleanness                 0.32
#
# LARGE_NUMBER                     :   VAST_AMOUNT                   (nn_weighted_adjective_noun_identity)
# SIZE                      0.67   :   QUANTITY                  0.58
# QUANTITY                  0.45   :   SIZE                      0.57
# WIDTH                     0.38   :   WIDTH                     0.42
# volume                    0.38   :   thickness                 0.38
# complexity                0.34   :   substantiality            0.35
#
# OLD_PERSON                       :   ELDERLY_LADY                  (nn_weighted_adjective_noun_identity)
# AGE                       0.45   :   commonness                0.31
# otherness                 0.40   :   AGE                       0.30
# typicality                0.40   :   health                    0.29
# abstractness              0.37   :   thoughtfulness            0.29
# originality               0.36   :   kindness                  0.29
#
# EARLIER_WORK                     :   EARLY_STAGE                   (nn_weighted_adjective_noun_identity)
# TIMING                    0.34   :   TIMING                    0.61
# regularity                0.33   :   height                    0.37
# accuracy                  0.33   :   maturity                  0.30
# otherness                 0.33   :   nature                    0.29
# ordinariness              0.32   :   seriousness               0.28
#
# GENERAL_PRINCIPLE                :   BASIC_RULE                    (nn_weighted_adjective_noun_identity)
# GENERALITY                0.58   :   GENERALITY                0.44
# concreteness              0.35   :   humaneness                0.43
# equality                  0.33   :   practicality              0.43
# humanness                 0.33   :   comprehensiveness         0.41
# morality                  0.33   :   reasonableness            0.41
#
# LARGE_QUANTITY                   :   GREAT_MAJORITY                (nn_weighted_adjective_noun_identity)
# quantity                  0.78   :   majority                  0.70
# SIZE                      0.63   :   quality                   0.41
# width                     0.40   :   minority                  0.41
# thickness                 0.38   :   SIZE                      0.32
# volume                    0.36   :   cleanness                 0.32
#
# NEW_LIFE                         :   EARLY_AGE                     (nn_weighted_adjective_noun_identity)
# freshness                 0.43   :   age                       0.62
# permanence                0.43   :   timing                    0.58
# individuality             0.43   :   height                    0.33
# originality               0.42   :   maturity                  0.32
# dullness                  0.42   :   duration                  0.32
#
# GENERAL_PRINCIPLE                :   BASIC_RULE                    (nn_weighted_adjective_noun_identity)
# GENERALITY                0.58   :   GENERALITY                0.44
# concreteness              0.35   :   humaneness                0.43
# equality                  0.33   :   practicality              0.43
# humanness                 0.33   :   comprehensiveness         0.41
# morality                  0.33   :   reasonableness            0.41
#
# DIFFERENT_KIND                   :   VARIOUS_FORM                  (nn_weighted_adjective_noun_identity)
# SAMENESS                  0.57   :   OTHERNESS                 0.39
# similarity                0.54   :   COMMONALITY               0.34
# COMMONALITY               0.50   :   foreignness               0.33
# DIFFERENCE                0.50   :   SAMENESS                  0.33
# OTHERNESS                 0.48   :   DIFFERENCE                0.33
#
# CERTAIN_CIRCUMSTANCE             :   PARTICULAR_CASE               (nn_weighted_adjective_noun_identity)
# COMMONNESS                0.42   :   significance              0.49
# otherness                 0.40   :   importance                0.45
# foreignness               0.39   :   COMMONNESS                0.43
# concreteness              0.37   :   abstractness              0.41
# ordinariness              0.37   :   seriousness               0.40
#
# IMPORTANT_PART                   :   SIGNIFICANT_ROLE              (nn_weighted_adjective_noun_identity)
# IMPORTANCE                0.62   :   SIGNIFICANCE              0.56
# SIGNIFICANCE              0.54   :   IMPORTANCE                0.53
# necessity                 0.41   :   status                    0.40
# pride                     0.39   :   RESPONSIBILITY            0.36
# RESPONSIBILITY            0.35   :   likelihood                0.36
#
# VAST_AMOUNT                      :   LARGE_QUANTITY                (nn_weighted_adjective_noun_identity)
# QUANTITY                  0.58   :   QUANTITY                  0.78
# SIZE                      0.57   :   SIZE                      0.63
# WIDTH                     0.42   :   WIDTH                     0.40
# THICKNESS                 0.38   :   THICKNESS                 0.38
# substantiality            0.35   :   volume                    0.36
#
# LARGE_NUMBER                     :   VAST_AMOUNT                   (nn_weighted_adjective_noun_identity)
# SIZE                      0.67   :   QUANTITY                  0.58
# QUANTITY                  0.45   :   SIZE                      0.57
# WIDTH                     0.38   :   WIDTH                     0.42
# volume                    0.38   :   thickness                 0.38
# complexity                0.34   :   substantiality            0.35
#
# GENERAL_PRINCIPLE                :   BASIC_RULE                    (nn_weighted_adjective_noun_identity)
# GENERALITY                0.58   :   GENERALITY                0.44
# concreteness              0.35   :   humaneness                0.43
# equality                  0.33   :   practicality              0.43
# humanness                 0.33   :   comprehensiveness         0.41
# morality                  0.33   :   reasonableness            0.41
#
# DIFFERENT_KIND                   :   VARIOUS_FORM                  (nn_weighted_adjective_noun_identity)
# SAMENESS                  0.57   :   OTHERNESS                 0.39
# similarity                0.54   :   COMMONALITY               0.34
# COMMONALITY               0.50   :   foreignness               0.33
# DIFFERENCE                0.50   :   SAMENESS                  0.33
# OTHERNESS                 0.48   :   DIFFERENCE                0.33
#
# FEDERAL_ASSEMBLY                 :   NATIONAL_GOVERNMENT           (nn_weighted_adjective_noun_identity)
# majority                  0.23   :   importance                0.35
# otherness                 0.22   :   significance              0.31
# nature                    0.22   :   stature                   0.30
# seniority                 0.21   :   seriousness               0.29
# sameness                  0.21   :   intrusiveness             0.29
#
# LARGE_NUMBER                     :   VAST_AMOUNT                   (nn_weighted_adjective_noun_identity)
# SIZE                      0.67   :   QUANTITY                  0.58
# QUANTITY                  0.45   :   SIZE                      0.57
# WIDTH                     0.38   :   WIDTH                     0.42
# volume                    0.38   :   thickness                 0.38
# complexity                0.34   :   substantiality            0.35
#
# OLD_PERSON                       :   ELDERLY_LADY                  (nn_weighted_adjective_noun_identity)
# AGE                       0.45   :   commonness                0.31
# otherness                 0.40   :   AGE                       0.30
# typicality                0.40   :   health                    0.29
# abstractness              0.37   :   thoughtfulness            0.29
# originality               0.36   :   kindness                  0.29
#
# NEW_INFORMATION                  :   FURTHER_EVIDENCE              (nn_weighted_adjective_noun_identity)
# individuality             0.46   :   possibility               0.36
# originality               0.44   :   finality                  0.36
# permanence                0.44   :   credibility               0.36
# immediacy                 0.44   :   sharpness                 0.34
# correctness               0.42   :   freshness                 0.34
#
# SPECIAL_CIRCUMSTANCE             :   PARTICULAR_CASE               (nn_weighted_adjective_noun_identity)
# ordinariness              0.47   :   significance              0.49
# COMMONNESS                0.44   :   importance                0.45
# otherness                 0.40   :   COMMONNESS                0.43
# nobility                  0.37   :   abstractness              0.41
# gregariousness            0.37   :   seriousness               0.40
#
# VAST_AMOUNT                      :   LARGE_QUANTITY                (nn_weighted_adjective_noun_identity)
# QUANTITY                  0.58   :   QUANTITY                  0.78
# SIZE                      0.57   :   SIZE                      0.63
# WIDTH                     0.42   :   WIDTH                     0.40
# THICKNESS                 0.38   :   THICKNESS                 0.38
# substantiality            0.35   :   volume                    0.36
#
# VAST_AMOUNT                      :   LARGE_QUANTITY                (nn_weighted_adjective_noun_identity)
# QUANTITY                  0.58   :   QUANTITY                  0.78
# SIZE                      0.57   :   SIZE                      0.63
# WIDTH                     0.42   :   WIDTH                     0.40
# THICKNESS                 0.38   :   THICKNESS                 0.38
# substantiality            0.35   :   volume                    0.36
#
# NEW_INFORMATION                  :   FURTHER_EVIDENCE              (nn_weighted_adjective_noun_identity)
# individuality             0.46   :   possibility               0.36
# originality               0.44   :   finality                  0.36
# permanence                0.44   :   credibility               0.36
# immediacy                 0.44   :   sharpness                 0.34
# correctness               0.42   :   freshness                 0.34
#
# VAST_AMOUNT                      :   LARGE_QUANTITY                (nn_weighted_adjective_noun_identity)
# QUANTITY                  0.58   :   QUANTITY                  0.78
# SIZE                      0.57   :   SIZE                      0.63
# WIDTH                     0.42   :   WIDTH                     0.40
# THICKNESS                 0.38   :   THICKNESS                 0.38
# substantiality            0.35   :   volume                    0.36
#
# LARGE_NUMBER                     :   VAST_AMOUNT                   (nn_weighted_adjective_noun_identity)
# SIZE                      0.67   :   QUANTITY                  0.58
# QUANTITY                  0.45   :   SIZE                      0.57
# WIDTH                     0.38   :   WIDTH                     0.42
# volume                    0.38   :   thickness                 0.38
# complexity                0.34   :   substantiality            0.35
#
# OLD_PERSON                       :   ELDERLY_LADY                  (nn_weighted_adjective_noun_identity)
# AGE                       0.45   :   commonness                0.31
# otherness                 0.40   :   AGE                       0.30
# typicality                0.40   :   health                    0.29
# abstractness              0.37   :   thoughtfulness            0.29
# originality               0.36   :   kindness                  0.29
#
# DIFFERENT_KIND                   :   VARIOUS_FORM                  (nn_weighted_adjective_noun_identity)
# SAMENESS                  0.57   :   OTHERNESS                 0.39
# similarity                0.54   :   COMMONALITY               0.34
# COMMONALITY               0.50   :   foreignness               0.33
# DIFFERENCE                0.50   :   SAMENESS                  0.33
# OTHERNESS                 0.48   :   DIFFERENCE                0.33
#
# GENERAL_PRINCIPLE                :   BASIC_RULE                    (nn_weighted_adjective_noun_identity)
# GENERALITY                0.58   :   GENERALITY                0.44
# concreteness              0.35   :   humaneness                0.43
# equality                  0.33   :   practicality              0.43
# humanness                 0.33   :   comprehensiveness         0.41
# morality                  0.33   :   reasonableness            0.41
#
# LARGE_NUMBER                     :   VAST_AMOUNT                   (nn_weighted_adjective_noun_identity)
# SIZE                      0.67   :   QUANTITY                  0.58
# QUANTITY                  0.45   :   SIZE                      0.57
# WIDTH                     0.38   :   WIDTH                     0.42
# volume                    0.38   :   thickness                 0.38
# complexity                0.34   :   substantiality            0.35
#
# GENERAL_PRINCIPLE                :   BASIC_RULE                    (nn_weighted_adjective_noun_identity)
# GENERALITY                0.58   :   GENERALITY                0.44
# concreteness              0.35   :   humaneness                0.43
# equality                  0.33   :   practicality              0.43
# humanness                 0.33   :   comprehensiveness         0.41
# morality                  0.33   :   reasonableness            0.41
#
# LARGE_QUANTITY                   :   GREAT_MAJORITY                (nn_weighted_adjective_noun_identity)
# quantity                  0.78   :   majority                  0.70
# SIZE                      0.63   :   quality                   0.41
# width                     0.40   :   minority                  0.41
# thickness                 0.38   :   SIZE                      0.32
# volume                    0.36   :   cleanness                 0.32
#
# IMPORTANT_PART                   :   SIGNIFICANT_ROLE              (nn_weighted_adjective_noun_identity)
# IMPORTANCE                0.62   :   SIGNIFICANCE              0.56
# SIGNIFICANCE              0.54   :   IMPORTANCE                0.53
# necessity                 0.41   :   status                    0.40
# pride                     0.39   :   RESPONSIBILITY            0.36
# RESPONSIBILITY            0.35   :   likelihood                0.36
#
# NEW_INFORMATION                  :   FURTHER_EVIDENCE              (nn_weighted_adjective_noun_identity)
# individuality             0.46   :   possibility               0.36
# originality               0.44   :   finality                  0.36
# permanence                0.44   :   credibility               0.36
# immediacy                 0.44   :   sharpness                 0.34
# correctness               0.42   :   freshness                 0.34
#
# VAST_AMOUNT                      :   LARGE_QUANTITY                (nn_weighted_adjective_noun_identity)
# QUANTITY                  0.58   :   QUANTITY                  0.78
# SIZE                      0.57   :   SIZE                      0.63
# WIDTH                     0.42   :   WIDTH                     0.40
# THICKNESS                 0.38   :   THICKNESS                 0.38
# substantiality            0.35   :   volume                    0.36
#
# LARGE_NUMBER                     :   GREAT_MAJORITY                (nn_weighted_adjective_noun_identity)
# SIZE                      0.67   :   majority                  0.70
# quantity                  0.45   :   quality                   0.41
# width                     0.38   :   minority                  0.41
# volume                    0.38   :   SIZE                      0.32
# complexity                0.34   :   cleanness                 0.32
#
# SPECIAL_CIRCUMSTANCE             :   PARTICULAR_CASE               (nn_weighted_adjective_noun_identity)
# ordinariness              0.47   :   significance              0.49
# COMMONNESS                0.44   :   importance                0.45
# otherness                 0.40   :   COMMONNESS                0.43
# nobility                  0.37   :   abstractness              0.41
# gregariousness            0.37   :   seriousness               0.40
#
# INDUSTRIAL_AREA                  :   WHOLE_COUNTRY                 (nn_weighted_adjective_noun_identity)
# CLEANNESS                 0.37   :   integrity                 0.51
# naturalness               0.32   :   originality               0.44
# tractability              0.31   :   quality                   0.44
# commerce                  0.31   :   evenness                  0.43
# pleasantness              0.30   :   CLEANNESS                 0.42
#
# SOCIAL_ACTIVITY                  :   ECONOMIC_CONDITION            (nn_weighted_adjective_noun_identity)
# sociability               0.45   :   health                    0.35
# sociality                 0.40   :   commerce                  0.33
# otherness                 0.32   :   importance                0.31
# domesticity               0.30   :   necessity                 0.31
# MORALITY                  0.30   :   MORALITY                  0.30
#
# GOOD_PLACE                       :   HIGH_POINT                    (nn_weighted_adjective_noun_identity)
# quality                   0.63   :   degree                    0.69
# cleanness                 0.48   :   height                    0.54
# consistency               0.48   :   distance                  0.29
# freshness                 0.46   :   age                       0.27
# solidity                  0.45   :   hardness                  0.27
#
# NEW_INFORMATION                  :   FURTHER_EVIDENCE              (nn_weighted_adjective_noun_identity)
# individuality             0.46   :   possibility               0.36
# originality               0.44   :   finality                  0.36
# permanence                0.44   :   credibility               0.36
# immediacy                 0.44   :   sharpness                 0.34
# correctness               0.42   :   freshness                 0.34
#
# DIFFERENT_PART                   :   VARIOUS_FORM                  (nn_weighted_adjective_noun_identity)
# DIFFERENCE                0.56   :   otherness                 0.39
# similarity                0.51   :   COMMONALITY               0.34
# COMMONALITY               0.50   :   foreignness               0.33
# SAMENESS                  0.45   :   SAMENESS                  0.33
# individuality             0.42   :   DIFFERENCE                0.33
#
# ECONOMIC_PROBLEM                 :   PRACTICAL_DIFFICULTY          (nn_weighted_adjective_noun_identity)
# crisis                    0.40   :   difficulty                0.71
# commerce                  0.36   :   PRACTICALITY              0.59
# moderation                0.33   :   COMPLEXITY                0.49
# PRACTICALITY              0.32   :   necessity                 0.37
# COMPLEXITY                0.31   :   importance                0.35
#
# EFFECTIVE_WAY                    :   IMPORTANT_PART                (nn_weighted_adjective_noun_identity)
# effectiveness             0.75   :   importance                0.62
# efficacy                  0.48   :   significance              0.54
# potency                   0.44   :   necessity                 0.41
# practicality              0.43   :   pride                     0.39
# accuracy                  0.41   :   responsibility            0.35
#
# NEW_SITUATION                    :   DIFFERENT_KIND                (nn_weighted_adjective_noun_identity)
# difficulty                0.47   :   sameness                  0.57
# complexity                0.43   :   similarity                0.54
# abstractness              0.42   :   commonality               0.50
# concreteness              0.40   :   difference                0.50
# necessity                 0.39   :   otherness                 0.48
#
# LARGE_NUMBER                     :   GREAT_MAJORITY                (nn_weighted_adjective_noun_identity)
# SIZE                      0.67   :   majority                  0.70
# quantity                  0.45   :   quality                   0.41
# width                     0.38   :   minority                  0.41
# volume                    0.38   :   SIZE                      0.32
# complexity                0.34   :   cleanness                 0.32
#
# CERTAIN_CIRCUMSTANCE             :   PARTICULAR_CASE               (nn_weighted_adjective_noun_identity)
# COMMONNESS                0.42   :   significance              0.49
# otherness                 0.40   :   importance                0.45
# foreignness               0.39   :   COMMONNESS                0.43
# concreteness              0.37   :   abstractness              0.41
# ordinariness              0.37   :   seriousness               0.40
#
# VAST_AMOUNT                      :   HIGH_PRICE                    (nn_weighted_adjective_noun_identity)
# QUANTITY                  0.58   :   degree                    0.54
# size                      0.57   :   height                    0.48
# width                     0.42   :   price                     0.46
# thickness                 0.38   :   moderation                0.44
# substantiality            0.35   :   QUANTITY                  0.33
#
# NEW_LAW                          :   BASIC_RULE                    (nn_weighted_adjective_noun_identity)
# conventionality           0.35   :   generality                0.44
# abstractness              0.34   :   humaneness                0.43
# originality               0.34   :   practicality              0.43
# liveliness                0.33   :   comprehensiveness         0.41
# individuality             0.33   :   reasonableness            0.41
#
# CENTRAL_AUTHORITY                :   LOCAL_OFFICE                  (nn_weighted_adjective_noun_identity)
# importance                0.39   :   handiness                 0.33
# significance              0.38   :   cleanness                 0.30
# power                     0.32   :   immediacy                 0.30
# potency                   0.31   :   freshness                 0.30
# responsibility            0.30   :   friendliness              0.29
#
# LARGE_NUMBER                     :   GREAT_MAJORITY                (nn_weighted_adjective_noun_identity)
# SIZE                      0.67   :   majority                  0.70
# quantity                  0.45   :   quality                   0.41
# width                     0.38   :   minority                  0.41
# volume                    0.38   :   SIZE                      0.32
# complexity                0.34   :   cleanness                 0.32
#
# OLD_PERSON                       :   ELDERLY_LADY                  (nn_weighted_adjective_noun_identity)
# AGE                       0.45   :   commonness                0.31
# otherness                 0.40   :   AGE                       0.30
# typicality                0.40   :   health                    0.29
# abstractness              0.37   :   thoughtfulness            0.29
# originality               0.36   :   kindness                  0.29
#
# GENERAL_PRINCIPLE                :   BASIC_RULE                    (nn_weighted_adjective_noun_identity)
# GENERALITY                0.58   :   GENERALITY                0.44
# concreteness              0.35   :   humaneness                0.43
# equality                  0.33   :   practicality              0.43
# humanness                 0.33   :   comprehensiveness         0.41
# morality                  0.33   :   reasonableness            0.41
#
# LARGE_NUMBER                     :   GREAT_MAJORITY                (nn_weighted_adjective_noun_identity)
# SIZE                      0.67   :   majority                  0.70
# quantity                  0.45   :   quality                   0.41
# width                     0.38   :   minority                  0.41
# volume                    0.38   :   SIZE                      0.32
# complexity                0.34   :   cleanness                 0.32
#
# CERTAIN_CIRCUMSTANCE             :   PARTICULAR_CASE               (nn_weighted_adjective_noun_identity)
# COMMONNESS                0.42   :   significance              0.49
# otherness                 0.40   :   importance                0.45
# foreignness               0.39   :   COMMONNESS                0.43
# concreteness              0.37   :   abstractness              0.41
# ordinariness              0.37   :   seriousness               0.40
#
# PREVIOUS_DAY                     :   EARLY_AGE                     (nn_weighted_adjective_noun_identity)
# ordinariness              0.46   :   age                       0.62
# conventionality           0.43   :   TIMING                    0.58
# sameness                  0.42   :   height                    0.33
# TIMING                    0.42   :   maturity                  0.32
# similarity                0.42   :   duration                  0.32
#
# LARGE_NUMBER                     :   VAST_AMOUNT                   (nn_weighted_adjective_noun_identity)
# SIZE                      0.67   :   QUANTITY                  0.58
# QUANTITY                  0.45   :   SIZE                      0.57
# WIDTH                     0.38   :   WIDTH                     0.42
# volume                    0.38   :   thickness                 0.38
# complexity                0.34   :   substantiality            0.35
#
# EFFECTIVE_WAY                    :   EFFICIENT_USE                 (nn_weighted_adjective_noun_identity)
# EFFECTIVENESS             0.75   :   EFFECTIVENESS             0.48
# efficacy                  0.48   :   PRACTICALITY              0.39
# potency                   0.44   :   ACCURACY                  0.38
# PRACTICALITY              0.43   :   quality                   0.34
# ACCURACY                  0.41   :   utility                   0.34
#
# LARGE_NUMBER                     :   GREAT_MAJORITY                (nn_weighted_adjective_noun_identity)
# SIZE                      0.67   :   majority                  0.70
# quantity                  0.45   :   quality                   0.41
# width                     0.38   :   minority                  0.41
# volume                    0.38   :   SIZE                      0.32
# complexity                0.34   :   cleanness                 0.32
#
# RIGHT_HAND                       :   LEFT_ARM                      (nn_weighted_adjective_noun_identity)
# POSITION                  0.62   :   POSITION                  0.48
# correctness               0.44   :   timing                    0.33
# rightness                 0.36   :   power                     0.31
# propriety                 0.34   :   length                    0.29
# direction                 0.34   :   strength                  0.25
#
# OLD_PERSON                       :   ELDERLY_LADY                  (nn_weighted_adjective_noun_identity)
# AGE                       0.45   :   commonness                0.31
# otherness                 0.40   :   AGE                       0.30
# typicality                0.40   :   health                    0.29
# abstractness              0.37   :   thoughtfulness            0.29
# originality               0.36   :   kindness                  0.29
#
# FEDERAL_ASSEMBLY                 :   NATIONAL_GOVERNMENT           (nn_weighted_adjective_noun_identity)
# majority                  0.23   :   importance                0.35
# otherness                 0.22   :   significance              0.31
# nature                    0.22   :   stature                   0.30
# seniority                 0.21   :   seriousness               0.29
# sameness                  0.21   :   intrusiveness             0.29
#
# BLACK_HAIR                       :   DARK_EYE                      (nn_weighted_adjective_noun_identity)
# evil                      0.45   :   luminosity                0.48
# length                    0.37   :   light                     0.43
# morality                  0.34   :   evenness                  0.37
# texture                   0.32   :   opacity                   0.35
# complexion                0.32   :   sharpness                 0.34
#
# SMALL_HOUSE                      :   LITTLE_ROOM                   (nn_weighted_adjective_noun_identity)
# SIZE                      0.77   :   SIZE                      0.48
# HEIGHT                    0.50   :   HEIGHT                    0.44
# width                     0.46   :   niceness                  0.42
# length                    0.39   :   cleanness                 0.42
# stature                   0.39   :   hardness                  0.41
#
# AMERICAN_COUNTRY                 :   EUROPEAN_STATE                (nn_weighted_adjective_noun_identity)
# foreignness               0.32   :   potency                   0.23
# corruptness               0.27   :   solidity                  0.20
# cleanness                 0.25   :   typicality                0.20
# PURITY                    0.25   :   utility                   0.19
# niceness                  0.25   :   PURITY                    0.19
#
# NEW_INFORMATION                  :   FURTHER_EVIDENCE              (nn_weighted_adjective_noun_identity)
# individuality             0.46   :   possibility               0.36
# originality               0.44   :   finality                  0.36
# permanence                0.44   :   credibility               0.36
# immediacy                 0.44   :   sharpness                 0.34
# correctness               0.42   :   freshness                 0.34
#
# SPECIAL_CIRCUMSTANCE             :   PARTICULAR_CASE               (nn_weighted_adjective_noun_identity)
# ordinariness              0.47   :   significance              0.49
# COMMONNESS                0.44   :   importance                0.45
# otherness                 0.40   :   COMMONNESS                0.43
# nobility                  0.37   :   abstractness              0.41
# gregariousness            0.37   :   seriousness               0.40
#
# DIFFERENT_PART                   :   VARIOUS_FORM                  (nn_weighted_adjective_noun_identity)
# DIFFERENCE                0.56   :   otherness                 0.39
# similarity                0.51   :   COMMONALITY               0.34
# COMMONALITY               0.50   :   foreignness               0.33
# SAMENESS                  0.45   :   SAMENESS                  0.33
# individuality             0.42   :   DIFFERENCE                0.33
#
# SPECIAL_CIRCUMSTANCE             :   PARTICULAR_CASE               (nn_weighted_adjective_noun_identity)
# ordinariness              0.47   :   significance              0.49
# COMMONNESS                0.44   :   importance                0.45
# otherness                 0.40   :   COMMONNESS                0.43
# nobility                  0.37   :   abstractness              0.41
# gregariousness            0.37   :   seriousness               0.40
#
# NEW_SITUATION                    :   DIFFERENT_KIND                (nn_weighted_adjective_noun_identity)
# difficulty                0.47   :   sameness                  0.57
# complexity                0.43   :   similarity                0.54
# abstractness              0.42   :   commonality               0.50
# concreteness              0.40   :   difference                0.50
# necessity                 0.39   :   otherness                 0.48
#
# OLD_PERSON                       :   ELDERLY_LADY                  (nn_weighted_adjective_noun_identity)
# AGE                       0.45   :   commonness                0.31
# otherness                 0.40   :   AGE                       0.30
# typicality                0.40   :   health                    0.29
# abstractness              0.37   :   thoughtfulness            0.29
# originality               0.36   :   kindness                  0.29
#
# EARLIER_WORK                     :   EARLY_STAGE                   (nn_weighted_adjective_noun_identity)
# TIMING                    0.34   :   TIMING                    0.61
# regularity                0.33   :   height                    0.37
# accuracy                  0.33   :   maturity                  0.30
# otherness                 0.33   :   nature                    0.29
# ordinariness              0.32   :   seriousness               0.28
#
# DIFFERENT_KIND                   :   VARIOUS_FORM                  (nn_weighted_adjective_noun_identity)
# SAMENESS                  0.57   :   OTHERNESS                 0.39
# similarity                0.54   :   COMMONALITY               0.34
# COMMONALITY               0.50   :   foreignness               0.33
# DIFFERENCE                0.50   :   SAMENESS                  0.33
# OTHERNESS                 0.48   :   DIFFERENCE                0.33
#
# EFFECTIVE_WAY                    :   EFFICIENT_USE                 (nn_weighted_adjective_noun_identity)
# EFFECTIVENESS             0.75   :   EFFECTIVENESS             0.48
# efficacy                  0.48   :   PRACTICALITY              0.39
# potency                   0.44   :   ACCURACY                  0.38
# PRACTICALITY              0.43   :   quality                   0.34
# ACCURACY                  0.41   :   utility                   0.34
#
# MAJOR_ISSUE                      :   AMERICAN_COUNTRY              (nn_weighted_adjective_noun_identity)
# significance              0.46   :   FOREIGNNESS               0.32
# importance                0.44   :   corruptness               0.27
# size                      0.35   :   cleanness                 0.25
# stature                   0.34   :   purity                    0.25
# FOREIGNNESS               0.32   :   niceness                  0.25
#
# LARGE_QUANTITY                   :   GREAT_MAJORITY                (nn_weighted_adjective_noun_identity)
# quantity                  0.78   :   majority                  0.70
# SIZE                      0.63   :   quality                   0.41
# width                     0.40   :   minority                  0.41
# thickness                 0.38   :   SIZE                      0.32
# volume                    0.36   :   cleanness                 0.32
#
# BLACK_HAIR                       :   DARK_EYE                      (nn_weighted_adjective_noun_identity)
# evil                      0.45   :   luminosity                0.48
# length                    0.37   :   light                     0.43
# morality                  0.34   :   evenness                  0.37
# texture                   0.32   :   opacity                   0.35
# complexion                0.32   :   sharpness                 0.34
#
# VAST_AMOUNT                      :   LARGE_QUANTITY                (nn_weighted_adjective_noun_identity)
# QUANTITY                  0.58   :   QUANTITY                  0.78
# SIZE                      0.57   :   SIZE                      0.63
# WIDTH                     0.42   :   WIDTH                     0.40
# THICKNESS                 0.38   :   THICKNESS                 0.38
# substantiality            0.35   :   volume                    0.36
#
# EARLIER_WORK                     :   EARLY_STAGE                   (nn_weighted_adjective_noun_identity)
# TIMING                    0.34   :   TIMING                    0.61
# regularity                0.33   :   height                    0.37
# accuracy                  0.33   :   maturity                  0.30
# otherness                 0.33   :   nature                    0.29
# ordinariness              0.32   :   seriousness               0.28
#
# GENERAL_PRINCIPLE                :   BASIC_RULE                    (nn_weighted_adjective_noun_identity)
# GENERALITY                0.58   :   GENERALITY                0.44
# concreteness              0.35   :   humaneness                0.43
# equality                  0.33   :   practicality              0.43
# humanness                 0.33   :   comprehensiveness         0.41
# morality                  0.33   :   reasonableness            0.41
#
# EFFECTIVE_WAY                    :   EFFICIENT_USE                 (nn_weighted_adjective_noun_identity)
# EFFECTIVENESS             0.75   :   EFFECTIVENESS             0.48
# efficacy                  0.48   :   PRACTICALITY              0.39
# potency                   0.44   :   ACCURACY                  0.38
# PRACTICALITY              0.43   :   quality                   0.34
# ACCURACY                  0.41   :   utility                   0.34
#
# OLDER_MAN                        :   ELDERLY_WOMAN                 (nn_weighted_adjective_noun_identity)
# AGE                       0.56   :   AGE                       0.39
# playfulness               0.40   :   fertility                 0.34
# emotionality              0.38   :   health                    0.32
# sociability               0.38   :   COMMONNESS                0.32
# COMMONNESS                0.38   :   equality                  0.30
#
# FEDERAL_ASSEMBLY                 :   NATIONAL_GOVERNMENT           (nn_weighted_adjective_noun_identity)
# majority                  0.23   :   importance                0.35
# otherness                 0.22   :   significance              0.31
# nature                    0.22   :   stature                   0.30
# seniority                 0.21   :   seriousness               0.29
# sameness                  0.21   :   intrusiveness             0.29
#
# VAST_AMOUNT                      :   HIGH_PRICE                    (nn_weighted_adjective_noun_identity)
# QUANTITY                  0.58   :   degree                    0.54
# size                      0.57   :   height                    0.48
# width                     0.42   :   price                     0.46
# thickness                 0.38   :   moderation                0.44
# substantiality            0.35   :   QUANTITY                  0.33
#
# SOCIAL_ACTIVITY                  :   POLITICAL_ACTION              (nn_weighted_adjective_noun_identity)
# sociability               0.45   :   action                    0.50
# sociality                 0.40   :   civility                  0.34
# otherness                 0.32   :   moderation                0.32
# domesticity               0.30   :   nastiness                 0.29
# morality                  0.30   :   fairness                  0.29
#
# RIGHT_HAND                       :   LEFT_ARM                      (nn_weighted_adjective_noun_identity)
# POSITION                  0.62   :   POSITION                  0.48
# correctness               0.44   :   timing                    0.33
# rightness                 0.36   :   power                     0.31
# propriety                 0.34   :   length                    0.29
# direction                 0.34   :   strength                  0.25
#
# VAST_AMOUNT                      :   HIGH_PRICE                    (nn_weighted_adjective_noun_identity)
# QUANTITY                  0.58   :   degree                    0.54
# size                      0.57   :   height                    0.48
# width                     0.42   :   price                     0.46
# thickness                 0.38   :   moderation                0.44
# substantiality            0.35   :   QUANTITY                  0.33
#
# OLD_PERSON                       :   ELDERLY_LADY                  (nn_weighted_adjective_noun_identity)
# AGE                       0.45   :   commonness                0.31
# otherness                 0.40   :   AGE                       0.30
# typicality                0.40   :   health                    0.29
# abstractness              0.37   :   thoughtfulness            0.29
# originality               0.36   :   kindness                  0.29
#
# GENERAL_PRINCIPLE                :   BASIC_RULE                    (nn_weighted_adjective_noun_identity)
# GENERALITY                0.58   :   GENERALITY                0.44
# concreteness              0.35   :   humaneness                0.43
# equality                  0.33   :   practicality              0.43
# humanness                 0.33   :   comprehensiveness         0.41
# morality                  0.33   :   reasonableness            0.41
#
# DIFFERENT_KIND                   :   VARIOUS_FORM                  (nn_weighted_adjective_noun_identity)
# SAMENESS                  0.57   :   OTHERNESS                 0.39
# similarity                0.54   :   COMMONALITY               0.34
# COMMONALITY               0.50   :   foreignness               0.33
# DIFFERENCE                0.50   :   SAMENESS                  0.33
# OTHERNESS                 0.48   :   DIFFERENCE                0.33
#
# EFFECTIVE_WAY                    :   EFFICIENT_USE                 (nn_weighted_adjective_noun_identity)
# EFFECTIVENESS             0.75   :   EFFECTIVENESS             0.48
# efficacy                  0.48   :   PRACTICALITY              0.39
# potency                   0.44   :   ACCURACY                  0.38
# PRACTICALITY              0.43   :   quality                   0.34
# ACCURACY                  0.41   :   utility                   0.34
#
# LARGE_QUANTITY                   :   GREAT_MAJORITY                (nn_weighted_adjective_noun_identity)
# quantity                  0.78   :   majority                  0.70
# SIZE                      0.63   :   quality                   0.41
# width                     0.40   :   minority                  0.41
# thickness                 0.38   :   SIZE                      0.32
# volume                    0.36   :   cleanness                 0.32
#
# GOOD_PLACE                       :   HIGH_POINT                    (nn_weighted_adjective_noun_identity)
# quality                   0.63   :   degree                    0.69
# cleanness                 0.48   :   height                    0.54
# consistency               0.48   :   distance                  0.29
# freshness                 0.46   :   age                       0.27
# solidity                  0.45   :   hardness                  0.27
#
# NEW_INFORMATION                  :   FURTHER_EVIDENCE              (nn_weighted_adjective_noun_identity)
# individuality             0.46   :   possibility               0.36
# originality               0.44   :   finality                  0.36
# permanence                0.44   :   credibility               0.36
# immediacy                 0.44   :   sharpness                 0.34
# correctness               0.42   :   freshness                 0.34
#
# SOCIAL_ACTIVITY                  :   ECONOMIC_CONDITION            (nn_weighted_adjective_noun_identity)
# sociability               0.45   :   health                    0.35
# sociality                 0.40   :   commerce                  0.33
# otherness                 0.32   :   importance                0.31
# domesticity               0.30   :   necessity                 0.31
# MORALITY                  0.30   :   MORALITY                  0.30
#
# SPECIAL_CIRCUMSTANCE             :   PARTICULAR_CASE               (nn_weighted_adjective_noun_identity)
# ordinariness              0.47   :   significance              0.49
# COMMONNESS                0.44   :   importance                0.45
# otherness                 0.40   :   COMMONNESS                0.43
# nobility                  0.37   :   abstractness              0.41
# gregariousness            0.37   :   seriousness               0.40
#
# LARGE_NUMBER                     :   GREAT_MAJORITY                (nn_weighted_adjective_noun_identity)
# SIZE                      0.67   :   majority                  0.70
# quantity                  0.45   :   quality                   0.41
# width                     0.38   :   minority                  0.41
# volume                    0.38   :   SIZE                      0.32
# complexity                0.34   :   cleanness                 0.32
#
# IMPORTANT_PART                   :   SIGNIFICANT_ROLE              (nn_weighted_adjective_noun_identity)
# IMPORTANCE                0.62   :   SIGNIFICANCE              0.56
# SIGNIFICANCE              0.54   :   IMPORTANCE                0.53
# necessity                 0.41   :   status                    0.40
# pride                     0.39   :   RESPONSIBILITY            0.36
# RESPONSIBILITY            0.35   :   likelihood                0.36
#
# OLD_PERSON                       :   ELDERLY_LADY                  (nn_weighted_adjective_noun_identity)
# AGE                       0.45   :   commonness                0.31
# otherness                 0.40   :   AGE                       0.30
# typicality                0.40   :   health                    0.29
# abstractness              0.37   :   thoughtfulness            0.29
# originality               0.36   :   kindness                  0.29
#
# EARLIER_WORK                     :   EARLY_STAGE                   (nn_weighted_adjective_noun_identity)
# TIMING                    0.34   :   TIMING                    0.61
# regularity                0.33   :   height                    0.37
# accuracy                  0.33   :   maturity                  0.30
# otherness                 0.33   :   nature                    0.29
# ordinariness              0.32   :   seriousness               0.28
#
# GENERAL_PRINCIPLE                :   BASIC_RULE                    (nn_weighted_adjective_noun_identity)
# GENERALITY                0.58   :   GENERALITY                0.44
# concreteness              0.35   :   humaneness                0.43
# equality                  0.33   :   practicality              0.43
# humanness                 0.33   :   comprehensiveness         0.41
# morality                  0.33   :   reasonableness            0.41
#
# CERTAIN_CIRCUMSTANCE             :   PARTICULAR_CASE               (nn_weighted_adjective_noun_identity)
# COMMONNESS                0.42   :   significance              0.49
# otherness                 0.40   :   importance                0.45
# foreignness               0.39   :   COMMONNESS                0.43
# concreteness              0.37   :   abstractness              0.41
# ordinariness              0.37   :   seriousness               0.40
#
# VAST_AMOUNT                      :   HIGH_PRICE                    (nn_weighted_adjective_noun_identity)
# QUANTITY                  0.58   :   degree                    0.54
# size                      0.57   :   height                    0.48
# width                     0.42   :   price                     0.46
# thickness                 0.38   :   moderation                0.44
# substantiality            0.35   :   QUANTITY                  0.33
#
# IMPORTANT_PART                   :   SIGNIFICANT_ROLE              (nn_weighted_adjective_noun_identity)
# IMPORTANCE                0.62   :   SIGNIFICANCE              0.56
# SIGNIFICANCE              0.54   :   IMPORTANCE                0.53
# necessity                 0.41   :   status                    0.40
# pride                     0.39   :   RESPONSIBILITY            0.36
# RESPONSIBILITY            0.35   :   likelihood                0.36
#
# NEW_LAW                          :   MODERN_LANGUAGE               (nn_weighted_adjective_noun_identity)
# CONVENTIONALITY           0.35   :   modernity                 0.60
# ABSTRACTNESS              0.34   :   foreignness               0.51
# originality               0.34   :   CONVENTIONALITY           0.48
# liveliness                0.33   :   ABSTRACTNESS              0.47
# individuality             0.33   :   otherness                 0.46
#
# LARGE_NUMBER                     :   VAST_AMOUNT                   (nn_weighted_adjective_noun_identity)
# SIZE                      0.67   :   QUANTITY                  0.58
# QUANTITY                  0.45   :   SIZE                      0.57
# WIDTH                     0.38   :   WIDTH                     0.42
# volume                    0.38   :   thickness                 0.38
# complexity                0.34   :   substantiality            0.35
#
# OLD_PERSON                       :   ELDERLY_LADY                  (nn_weighted_adjective_noun_identity)
# AGE                       0.45   :   commonness                0.31
# otherness                 0.40   :   AGE                       0.30
# typicality                0.40   :   health                    0.29
# abstractness              0.37   :   thoughtfulness            0.29
# originality               0.36   :   kindness                  0.29
#
# NEW_LIFE                         :   EARLY_AGE                     (nn_weighted_adjective_noun_identity)
# freshness                 0.43   :   age                       0.62
# permanence                0.43   :   timing                    0.58
# individuality             0.43   :   height                    0.33
# originality               0.42   :   maturity                  0.32
# dullness                  0.42   :   duration                  0.32
#
# POLITICAL_ACTION                 :   ECONOMIC_DEVELOPMENT          (nn_weighted_adjective_noun_identity)
# action                    0.50   :   commerce                  0.38
# civility                  0.34   :   completeness              0.29
# moderation                0.32   :   auspiciousness            0.28
# nastiness                 0.29   :   practicality              0.28
# fairness                  0.29   :   attractiveness            0.28
#
# LARGE_NUMBER                     :   GREAT_MAJORITY                (nn_weighted_adjective_noun_identity)
# SIZE                      0.67   :   majority                  0.70
# quantity                  0.45   :   quality                   0.41
# width                     0.38   :   minority                  0.41
# volume                    0.38   :   SIZE                      0.32
# complexity                0.34   :   cleanness                 0.32
#
# CERTAIN_CIRCUMSTANCE             :   PARTICULAR_CASE               (nn_weighted_adjective_noun_identity)
# COMMONNESS                0.42   :   significance              0.49
# otherness                 0.40   :   importance                0.45
# foreignness               0.39   :   COMMONNESS                0.43
# concreteness              0.37   :   abstractness              0.41
# ordinariness              0.37   :   seriousness               0.40
#
# CERTAIN_CIRCUMSTANCE             :   PARTICULAR_CASE               (nn_weighted_adjective_noun_identity)
# COMMONNESS                0.42   :   significance              0.49
# otherness                 0.40   :   importance                0.45
# foreignness               0.39   :   COMMONNESS                0.43
# concreteness              0.37   :   abstractness              0.41
# ordinariness              0.37   :   seriousness               0.40
#
# IMPORTANT_PART                   :   SIGNIFICANT_ROLE              (nn_weighted_adjective_noun_identity)
# IMPORTANCE                0.62   :   SIGNIFICANCE              0.56
# SIGNIFICANCE              0.54   :   IMPORTANCE                0.53
# necessity                 0.41   :   status                    0.40
# pride                     0.39   :   RESPONSIBILITY            0.36
# RESPONSIBILITY            0.35   :   likelihood                0.36
#
# DIFFERENT_KIND                   :   VARIOUS_FORM                  (nn_weighted_adjective_noun_identity)
# SAMENESS                  0.57   :   OTHERNESS                 0.39
# similarity                0.54   :   COMMONALITY               0.34
# COMMONALITY               0.50   :   foreignness               0.33
# DIFFERENCE                0.50   :   SAMENESS                  0.33
# OTHERNESS                 0.48   :   DIFFERENCE                0.33
#
# EFFECTIVE_WAY                    :   EFFICIENT_USE                 (nn_weighted_adjective_noun_identity)
# EFFECTIVENESS             0.75   :   EFFECTIVENESS             0.48
# efficacy                  0.48   :   PRACTICALITY              0.39
# potency                   0.44   :   ACCURACY                  0.38
# PRACTICALITY              0.43   :   quality                   0.34
# ACCURACY                  0.41   :   utility                   0.34
#
# OLDER_MAN                        :   ELDERLY_WOMAN                 (nn_weighted_adjective_noun_identity)
# AGE                       0.56   :   AGE                       0.39
# playfulness               0.40   :   fertility                 0.34
# emotionality              0.38   :   health                    0.32
# sociability               0.38   :   COMMONNESS                0.32
# COMMONNESS                0.38   :   equality                  0.30
#
# FEDERAL_ASSEMBLY                 :   NATIONAL_GOVERNMENT           (nn_weighted_adjective_noun_identity)
# majority                  0.23   :   importance                0.35
# otherness                 0.22   :   significance              0.31
# nature                    0.22   :   stature                   0.30
# seniority                 0.21   :   seriousness               0.29
# sameness                  0.21   :   intrusiveness             0.29
#
# LARGE_QUANTITY                   :   GREAT_MAJORITY                (nn_weighted_adjective_noun_identity)
# quantity                  0.78   :   majority                  0.70
# SIZE                      0.63   :   quality                   0.41
# width                     0.40   :   minority                  0.41
# thickness                 0.38   :   SIZE                      0.32
# volume                    0.36   :   cleanness                 0.32
#
# VAST_AMOUNT                      :   LARGE_QUANTITY                (nn_weighted_adjective_noun_identity)
# QUANTITY                  0.58   :   QUANTITY                  0.78
# SIZE                      0.57   :   SIZE                      0.63
# WIDTH                     0.42   :   WIDTH                     0.40
# THICKNESS                 0.38   :   THICKNESS                 0.38
# substantiality            0.35   :   volume                    0.36
#
# SPECIAL_CIRCUMSTANCE             :   PARTICULAR_CASE               (nn_weighted_adjective_noun_identity)
# ordinariness              0.47   :   significance              0.49
# COMMONNESS                0.44   :   importance                0.45
# otherness                 0.40   :   COMMONNESS                0.43
# nobility                  0.37   :   abstractness              0.41
# gregariousness            0.37   :   seriousness               0.40
#
# LOCAL_OFFICE                     :   NEW_TECHNOLOGY                (nn_weighted_adjective_noun_identity)
# handiness                 0.33   :   complexity                0.39
# cleanness                 0.30   :   FRESHNESS                 0.38
# immediacy                 0.30   :   practicality              0.38
# FRESHNESS                 0.30   :   capability                0.37
# friendliness              0.29   :   convenience               0.37
#
# CERTAIN_CIRCUMSTANCE             :   PARTICULAR_CASE               (nn_weighted_adjective_noun_identity)
# COMMONNESS                0.42   :   significance              0.49
# otherness                 0.40   :   importance                0.45
# foreignness               0.39   :   COMMONNESS                0.43
# concreteness              0.37   :   abstractness              0.41
# ordinariness              0.37   :   seriousness               0.40
#
# VAST_AMOUNT                      :   HIGH_PRICE                    (nn_weighted_adjective_noun_identity)
# QUANTITY                  0.58   :   degree                    0.54
# size                      0.57   :   height                    0.48
# width                     0.42   :   price                     0.46
# thickness                 0.38   :   moderation                0.44
# substantiality            0.35   :   QUANTITY                  0.33
#
# IMPORTANT_PART                   :   SIGNIFICANT_ROLE              (nn_weighted_adjective_noun_identity)
# IMPORTANCE                0.62   :   SIGNIFICANCE              0.56
# SIGNIFICANCE              0.54   :   IMPORTANCE                0.53
# necessity                 0.41   :   status                    0.40
# pride                     0.39   :   RESPONSIBILITY            0.36
# RESPONSIBILITY            0.35   :   likelihood                0.36
#
# EARLIER_WORK                     :   EARLY_STAGE                   (nn_weighted_adjective_noun_identity)
# TIMING                    0.34   :   TIMING                    0.61
# regularity                0.33   :   height                    0.37
# accuracy                  0.33   :   maturity                  0.30
# otherness                 0.33   :   nature                    0.29
# ordinariness              0.32   :   seriousness               0.28
#
# GENERAL_PRINCIPLE                :   BASIC_RULE                    (nn_weighted_adjective_noun_identity)
# GENERALITY                0.58   :   GENERALITY                0.44
# concreteness              0.35   :   humaneness                0.43
# equality                  0.33   :   practicality              0.43
# humanness                 0.33   :   comprehensiveness         0.41
# morality                  0.33   :   reasonableness            0.41
#
# LARGE_QUANTITY                   :   GREAT_MAJORITY                (nn_weighted_adjective_noun_identity)
# quantity                  0.78   :   majority                  0.70
# SIZE                      0.63   :   quality                   0.41
# width                     0.40   :   minority                  0.41
# thickness                 0.38   :   SIZE                      0.32
# volume                    0.36   :   cleanness                 0.32
#
# EFFECTIVE_WAY                    :   EFFICIENT_USE                 (nn_weighted_adjective_noun_identity)
# EFFECTIVENESS             0.75   :   EFFECTIVENESS             0.48
# efficacy                  0.48   :   PRACTICALITY              0.39
# potency                   0.44   :   ACCURACY                  0.38
# PRACTICALITY              0.43   :   quality                   0.34
# ACCURACY                  0.41   :   utility                   0.34
#
# OLDER_MAN                        :   ELDERLY_WOMAN                 (nn_weighted_adjective_noun_identity)
# AGE                       0.56   :   AGE                       0.39
# playfulness               0.40   :   fertility                 0.34
# emotionality              0.38   :   health                    0.32
# sociability               0.38   :   COMMONNESS                0.32
# COMMONNESS                0.38   :   equality                  0.30
#
# FEDERAL_ASSEMBLY                 :   NATIONAL_GOVERNMENT           (nn_weighted_adjective_noun_identity)
# majority                  0.23   :   importance                0.35
# otherness                 0.22   :   significance              0.31
# nature                    0.22   :   stature                   0.30
# seniority                 0.21   :   seriousness               0.29
# sameness                  0.21   :   intrusiveness             0.29
#
# LARGE_NUMBER                     :   VAST_AMOUNT                   (nn_weighted_adjective_noun_identity)
# SIZE                      0.67   :   QUANTITY                  0.58
# QUANTITY                  0.45   :   SIZE                      0.57
# WIDTH                     0.38   :   WIDTH                     0.42
# volume                    0.38   :   thickness                 0.38
# complexity                0.34   :   substantiality            0.35
#
# OLDER_MAN                        :   ELDERLY_WOMAN                 (nn_weighted_adjective_noun_identity)
# AGE                       0.56   :   AGE                       0.39
# playfulness               0.40   :   fertility                 0.34
# emotionality              0.38   :   health                    0.32
# sociability               0.38   :   COMMONNESS                0.32
# COMMONNESS                0.38   :   equality                  0.30
#
# FEDERAL_ASSEMBLY                 :   NATIONAL_GOVERNMENT           (nn_weighted_adjective_noun_identity)
# majority                  0.23   :   importance                0.35
# otherness                 0.22   :   significance              0.31
# nature                    0.22   :   stature                   0.30
# seniority                 0.21   :   seriousness               0.29
# sameness                  0.21   :   intrusiveness             0.29
#
# LARGE_NUMBER                     :   GREAT_MAJORITY                (nn_weighted_adjective_noun_identity)
# SIZE                      0.67   :   majority                  0.70
# quantity                  0.45   :   quality                   0.41
# width                     0.38   :   minority                  0.41
# volume                    0.38   :   SIZE                      0.32
# complexity                0.34   :   cleanness                 0.32
#
# CERTAIN_CIRCUMSTANCE             :   PARTICULAR_CASE               (nn_weighted_adjective_noun_identity)
# COMMONNESS                0.42   :   significance              0.49
# otherness                 0.40   :   importance                0.45
# foreignness               0.39   :   COMMONNESS                0.43
# concreteness              0.37   :   abstractness              0.41
# ordinariness              0.37   :   seriousness               0.40
#
# IMPORTANT_PART                   :   SIGNIFICANT_ROLE              (nn_weighted_adjective_noun_identity)
# IMPORTANCE                0.62   :   SIGNIFICANCE              0.56
# SIGNIFICANCE              0.54   :   IMPORTANCE                0.53
# necessity                 0.41   :   status                    0.40
# pride                     0.39   :   RESPONSIBILITY            0.36
# RESPONSIBILITY            0.35   :   likelihood                0.36
#
# LARGE_NUMBER                     :   VAST_AMOUNT                   (nn_weighted_adjective_noun_identity)
# SIZE                      0.67   :   QUANTITY                  0.58
# QUANTITY                  0.45   :   SIZE                      0.57
# WIDTH                     0.38   :   WIDTH                     0.42
# volume                    0.38   :   thickness                 0.38
# complexity                0.34   :   substantiality            0.35
#
# DIFFERENT_KIND                   :   VARIOUS_FORM                  (nn_weighted_adjective_noun_identity)
# SAMENESS                  0.57   :   OTHERNESS                 0.39
# similarity                0.54   :   COMMONALITY               0.34
# COMMONALITY               0.50   :   foreignness               0.33
# DIFFERENCE                0.50   :   SAMENESS                  0.33
# OTHERNESS                 0.48   :   DIFFERENCE                0.33
#
# OLDER_MAN                        :   ELDERLY_WOMAN                 (nn_weighted_adjective_noun_identity)
# AGE                       0.56   :   AGE                       0.39
# playfulness               0.40   :   fertility                 0.34
# emotionality              0.38   :   health                    0.32
# sociability               0.38   :   COMMONNESS                0.32
# COMMONNESS                0.38   :   equality                  0.30
#
# IMPORTANT_PART                   :   SIGNIFICANT_ROLE              (nn_weighted_adjective_noun_identity)
# IMPORTANCE                0.62   :   SIGNIFICANCE              0.56
# SIGNIFICANCE              0.54   :   IMPORTANCE                0.53
# necessity                 0.41   :   status                    0.40
# pride                     0.39   :   RESPONSIBILITY            0.36
# RESPONSIBILITY            0.35   :   likelihood                0.36
#
# OLD_PERSON                       :   ELDERLY_LADY                  (nn_weighted_adjective_noun_identity)
# AGE                       0.45   :   commonness                0.31
# otherness                 0.40   :   AGE                       0.30
# typicality                0.40   :   health                    0.29
# abstractness              0.37   :   thoughtfulness            0.29
# originality               0.36   :   kindness                  0.29
#
# EFFECTIVE_WAY                    :   EFFICIENT_USE                 (nn_weighted_adjective_noun_identity)
# EFFECTIVENESS             0.75   :   EFFECTIVENESS             0.48
# efficacy                  0.48   :   PRACTICALITY              0.39
# potency                   0.44   :   ACCURACY                  0.38
# PRACTICALITY              0.43   :   quality                   0.34
# ACCURACY                  0.41   :   utility                   0.34
#
# LARGE_NUMBER                     :   GREAT_MAJORITY                (nn_weighted_adjective_noun_identity)
# SIZE                      0.67   :   majority                  0.70
# quantity                  0.45   :   quality                   0.41
# width                     0.38   :   minority                  0.41
# volume                    0.38   :   SIZE                      0.32
# complexity                0.34   :   cleanness                 0.32
#
# CERTAIN_CIRCUMSTANCE             :   PARTICULAR_CASE               (nn_weighted_adjective_noun_identity)
# COMMONNESS                0.42   :   significance              0.49
# otherness                 0.40   :   importance                0.45
# foreignness               0.39   :   COMMONNESS                0.43
# concreteness              0.37   :   abstractness              0.41
# ordinariness              0.37   :   seriousness               0.40
#
# NEW_INFORMATION                  :   FURTHER_EVIDENCE              (nn_weighted_adjective_noun_identity)
# individuality             0.46   :   possibility               0.36
# originality               0.44   :   finality                  0.36
# permanence                0.44   :   credibility               0.36
# immediacy                 0.44   :   sharpness                 0.34
# correctness               0.42   :   freshness                 0.34
#
# VAST_AMOUNT                      :   LARGE_QUANTITY                (nn_weighted_adjective_noun_identity)
# QUANTITY                  0.58   :   QUANTITY                  0.78
# SIZE                      0.57   :   SIZE                      0.63
# WIDTH                     0.42   :   WIDTH                     0.40
# THICKNESS                 0.38   :   THICKNESS                 0.38
# substantiality            0.35   :   volume                    0.36
#
# SPECIAL_CIRCUMSTANCE             :   PARTICULAR_CASE               (nn_weighted_adjective_noun_identity)
# ordinariness              0.47   :   significance              0.49
# COMMONNESS                0.44   :   importance                0.45
# otherness                 0.40   :   COMMONNESS                0.43
# nobility                  0.37   :   abstractness              0.41
# gregariousness            0.37   :   seriousness               0.40
#
# GOOD_PLACE                       :   HIGH_POINT                    (nn_weighted_adjective_noun_identity)
# quality                   0.63   :   degree                    0.69
# cleanness                 0.48   :   height                    0.54
# consistency               0.48   :   distance                  0.29
# freshness                 0.46   :   age                       0.27
# solidity                  0.45   :   hardness                  0.27
#
# CERTAIN_CIRCUMSTANCE             :   PARTICULAR_CASE               (nn_weighted_adjective_noun_identity)
# COMMONNESS                0.42   :   significance              0.49
# otherness                 0.40   :   importance                0.45
# foreignness               0.39   :   COMMONNESS                0.43
# concreteness              0.37   :   abstractness              0.41
# ordinariness              0.37   :   seriousness               0.40
#
# VAST_AMOUNT                      :   HIGH_PRICE                    (nn_weighted_adjective_noun_identity)
# QUANTITY                  0.58   :   degree                    0.54
# size                      0.57   :   height                    0.48
# width                     0.42   :   price                     0.46
# thickness                 0.38   :   moderation                0.44
# substantiality            0.35   :   QUANTITY                  0.33
#
# NEW_LIFE                         :   ECONOMIC_DEVELOPMENT          (nn_weighted_adjective_noun_identity)
# freshness                 0.43   :   commerce                  0.38
# permanence                0.43   :   completeness              0.29
# individuality             0.43   :   auspiciousness            0.28
# originality               0.42   :   practicality              0.28
# dullness                  0.42   :   attractiveness            0.28
#
# LARGE_NUMBER                     :   VAST_AMOUNT                   (nn_weighted_adjective_noun_identity)
# SIZE                      0.67   :   QUANTITY                  0.58
# QUANTITY                  0.45   :   SIZE                      0.57
# WIDTH                     0.38   :   WIDTH                     0.42
# volume                    0.38   :   thickness                 0.38
# complexity                0.34   :   substantiality            0.35
#
# DIFFERENT_KIND                   :   VARIOUS_FORM                  (nn_weighted_adjective_noun_identity)
# SAMENESS                  0.57   :   OTHERNESS                 0.39
# similarity                0.54   :   COMMONALITY               0.34
# COMMONALITY               0.50   :   foreignness               0.33
# DIFFERENCE                0.50   :   SAMENESS                  0.33
# OTHERNESS                 0.48   :   DIFFERENCE                0.33
#
# FEDERAL_ASSEMBLY                 :   NATIONAL_GOVERNMENT           (nn_weighted_adjective_noun_identity)
# majority                  0.23   :   importance                0.35
# otherness                 0.22   :   significance              0.31
# nature                    0.22   :   stature                   0.30
# seniority                 0.21   :   seriousness               0.29
# sameness                  0.21   :   intrusiveness             0.29
#
# BLACK_HAIR                       :   DARK_EYE                      (nn_weighted_adjective_noun_identity)
# evil                      0.45   :   luminosity                0.48
# length                    0.37   :   light                     0.43
# morality                  0.34   :   evenness                  0.37
# texture                   0.32   :   opacity                   0.35
# complexion                0.32   :   sharpness                 0.34
#
# GOOD_PLACE                       :   HIGH_POINT                    (nn_weighted_adjective_noun_identity)
# quality                   0.63   :   degree                    0.69
# cleanness                 0.48   :   height                    0.54
# consistency               0.48   :   distance                  0.29
# freshness                 0.46   :   age                       0.27
# solidity                  0.45   :   hardness                  0.27
#
# NEW_INFORMATION                  :   FURTHER_EVIDENCE              (nn_weighted_adjective_noun_identity)
# individuality             0.46   :   possibility               0.36
# originality               0.44   :   finality                  0.36
# permanence                0.44   :   credibility               0.36
# immediacy                 0.44   :   sharpness                 0.34
# correctness               0.42   :   freshness                 0.34
#
# EFFECTIVE_WAY                    :   IMPORTANT_PART                (nn_weighted_adjective_noun_identity)
# effectiveness             0.75   :   importance                0.62
# efficacy                  0.48   :   significance              0.54
# potency                   0.44   :   necessity                 0.41
# practicality              0.43   :   pride                     0.39
# accuracy                  0.41   :   responsibility            0.35
#
# BETTER_JOB                       :   GOOD_EFFECT                   (nn_weighted_adjective_noun_identity)
# ease                      0.41   :   QUALITY                   0.50
# QUALITY                   0.40   :   effectiveness             0.43
# difficulty                0.40   :   friendliness              0.37
# naturalness               0.40   :   worthiness                0.37
# perfection                0.39   :   freshness                 0.37
#
# NEW_SITUATION                    :   DIFFERENT_KIND                (nn_weighted_adjective_noun_identity)
# difficulty                0.47   :   sameness                  0.57
# complexity                0.43   :   similarity                0.54
# abstractness              0.42   :   commonality               0.50
# concreteness              0.40   :   difference                0.50
# necessity                 0.39   :   otherness                 0.48
#
# ECONOMIC_DEVELOPMENT             :   RURAL_COMMUNITY               (nn_weighted_adjective_noun_identity)
# commerce                  0.38   :   literacy                  0.35
# completeness              0.29   :   PRACTICALITY              0.34
# auspiciousness            0.28   :   importance                0.31
# PRACTICALITY              0.28   :   ordinariness              0.31
# attractiveness            0.28   :   cleanness                 0.29
#
# GENERAL_LEVEL                    :   FEDERAL_ASSEMBLY              (nn_weighted_adjective_noun_identity)
# generality                0.56   :   majority                  0.23
# degree                    0.30   :   otherness                 0.22
# concreteness              0.30   :   nature                    0.22
# quality                   0.28   :   seniority                 0.21
# cleanness                 0.28   :   sameness                  0.21
#
# GENERAL_PRINCIPLE                :   PRESENT_POSITION              (nn_weighted_adjective_noun_identity)
# generality                0.58   :   position                  0.61
# concreteness              0.35   :   presence                  0.38
# equality                  0.33   :   strength                  0.31
# humanness                 0.33   :   otherness                 0.31
# morality                  0.33   :   stature                   0.30
#
# AMERICAN_COUNTRY                 :   EUROPEAN_STATE                (nn_weighted_adjective_noun_identity)
# foreignness               0.32   :   potency                   0.23
# corruptness               0.27   :   solidity                  0.20
# cleanness                 0.25   :   typicality                0.20
# PURITY                    0.25   :   utility                   0.19
# niceness                  0.25   :   PURITY                    0.19
#
# EARLY_STAGE                      :   LONG_PERIOD                   (nn_weighted_adjective_noun_identity)
# timing                    0.61   :   duration                  0.69
# height                    0.37   :   length                    0.41
# maturity                  0.30   :   distance                  0.36
# nature                    0.29   :   cyclicity                 0.35
# seriousness               0.28   :   fullness                  0.31
#
# CENTRAL_AUTHORITY                :   POLITICAL_ACTION              (nn_weighted_adjective_noun_identity)
# importance                0.39   :   action                    0.50
# significance              0.38   :   civility                  0.34
# power                     0.32   :   moderation                0.32
# potency                   0.31   :   nastiness                 0.29
# responsibility            0.30   :   fairness                  0.29
#
# EARLY_EVENING                    :   PREVIOUS_DAY                  (nn_weighted_adjective_noun_identity)
# TIMING                    0.70   :   ordinariness              0.46
# duration                  0.31   :   conventionality           0.43
# accuracy                  0.31   :   sameness                  0.42
# completeness              0.31   :   TIMING                    0.42
# cyclicity                 0.29   :   similarity                0.42
#
# SPECIAL_CIRCUMSTANCE             :   PARTICULAR_CASE               (nn_weighted_adjective_noun_identity)
# ordinariness              0.47   :   significance              0.49
# COMMONNESS                0.44   :   importance                0.45
# otherness                 0.40   :   COMMONNESS                0.43
# nobility                  0.37   :   abstractness              0.41
# gregariousness            0.37   :   seriousness               0.40
#
# EFFICIENT_USE                    :   SIGNIFICANT_ROLE              (nn_weighted_adjective_noun_identity)
# effectiveness             0.48   :   significance              0.56
# practicality              0.39   :   importance                0.53
# accuracy                  0.38   :   status                    0.40
# quality                   0.34   :   responsibility            0.36
# utility                   0.34   :   likelihood                0.36
#
# ECONOMIC_CONDITION               :   AMERICAN_COUNTRY              (nn_weighted_adjective_noun_identity)
# health                    0.35   :   foreignness               0.32
# commerce                  0.33   :   corruptness               0.27
# importance                0.31   :   cleanness                 0.25
# necessity                 0.31   :   purity                    0.25
# morality                  0.30   :   niceness                  0.25
#
# EFFECTIVE_WAY                    :   PRACTICAL_DIFFICULTY          (nn_weighted_adjective_noun_identity)
# effectiveness             0.75   :   difficulty                0.71
# efficacy                  0.48   :   PRACTICALITY              0.59
# potency                   0.44   :   complexity                0.49
# PRACTICALITY              0.43   :   necessity                 0.37
# accuracy                  0.41   :   importance                0.35
#
# EFFICIENT_USE                    :   LITTLE_ROOM                   (nn_weighted_adjective_noun_identity)
# effectiveness             0.48   :   size                      0.48
# practicality              0.39   :   height                    0.44
# accuracy                  0.38   :   niceness                  0.42
# quality                   0.34   :   cleanness                 0.42
# utility                   0.34   :   hardness                  0.41
#
# OLD_PERSON                       :   ELDERLY_LADY                  (nn_weighted_adjective_noun_identity)
# AGE                       0.45   :   commonness                0.31
# otherness                 0.40   :   AGE                       0.30
# typicality                0.40   :   health                    0.29
# abstractness              0.37   :   thoughtfulness            0.29
# originality               0.36   :   kindness                  0.29
#
# EARLIER_WORK                     :   EARLY_STAGE                   (nn_weighted_adjective_noun_identity)
# TIMING                    0.34   :   TIMING                    0.61
# regularity                0.33   :   height                    0.37
# accuracy                  0.33   :   maturity                  0.30
# otherness                 0.33   :   nature                    0.29
# ordinariness              0.32   :   seriousness               0.28
#
# BETTER_JOB                       :   GOOD_PLACE                    (nn_weighted_adjective_noun_identity)
# ease                      0.41   :   QUALITY                   0.63
# QUALITY                   0.40   :   cleanness                 0.48
# difficulty                0.40   :   consistency               0.48
# naturalness               0.40   :   freshness                 0.46
# perfection                0.39   :   solidity                  0.45
#
# DARK_EYE                         :   LEFT_ARM                      (nn_weighted_adjective_noun_identity)
# luminosity                0.48   :   position                  0.48
# light                     0.43   :   timing                    0.33
# evenness                  0.37   :   power                     0.31
# opacity                   0.35   :   length                    0.29
# sharpness                 0.34   :   strength                  0.25
#
# DIFFERENT_KIND                   :   VARIOUS_FORM                  (nn_weighted_adjective_noun_identity)
# SAMENESS                  0.57   :   OTHERNESS                 0.39
# similarity                0.54   :   COMMONALITY               0.34
# COMMONALITY               0.50   :   foreignness               0.33
# DIFFERENCE                0.50   :   SAMENESS                  0.33
# OTHERNESS                 0.48   :   DIFFERENCE                0.33
#
# LARGE_QUANTITY                   :   GREAT_MAJORITY                (nn_weighted_adjective_noun_identity)
# quantity                  0.78   :   majority                  0.70
# SIZE                      0.63   :   quality                   0.41
# width                     0.40   :   minority                  0.41
# thickness                 0.38   :   SIZE                      0.32
# volume                    0.36   :   cleanness                 0.32
#
# OLDER_MAN                        :   ELDERLY_WOMAN                 (nn_weighted_adjective_noun_identity)
# AGE                       0.56   :   AGE                       0.39
# playfulness               0.40   :   fertility                 0.34
# emotionality              0.38   :   health                    0.32
# sociability               0.38   :   COMMONNESS                0.32
# COMMONNESS                0.38   :   equality                  0.30
#
# ECONOMIC_CONDITION               :   AMERICAN_COUNTRY              (nn_weighted_adjective_noun_identity)
# health                    0.35   :   foreignness               0.32
# commerce                  0.33   :   corruptness               0.27
# importance                0.31   :   cleanness                 0.25
# necessity                 0.31   :   purity                    0.25
# morality                  0.30   :   niceness                  0.25
#
# EARLIER_WORK                     :   EARLY_EVENING                 (nn_weighted_adjective_noun_identity)
# TIMING                    0.34   :   TIMING                    0.70
# regularity                0.33   :   duration                  0.31
# ACCURACY                  0.33   :   ACCURACY                  0.31
# otherness                 0.33   :   completeness              0.31
# ordinariness              0.32   :   cyclicity                 0.29
#
# PREVIOUS_DAY                     :   EARLY_AGE                     (nn_weighted_adjective_noun_identity)
# ordinariness              0.46   :   age                       0.62
# conventionality           0.43   :   TIMING                    0.58
# sameness                  0.42   :   height                    0.33
# TIMING                    0.42   :   maturity                  0.32
# similarity                0.42   :   duration                  0.32
#
# PUBLIC_BUILDING                  :   CENTRAL_AUTHORITY             (nn_weighted_adjective_noun_identity)
# ordinariness              0.35   :   importance                0.39
# modesty                   0.33   :   significance              0.38
# height                    0.33   :   power                     0.32
# humanness                 0.33   :   potency                   0.31
# morality                  0.33   :   responsibility            0.30
#
# DIFFERENT_KIND                   :   VARIOUS_FORM                  (nn_weighted_adjective_noun_identity)
# SAMENESS                  0.57   :   OTHERNESS                 0.39
# similarity                0.54   :   COMMONALITY               0.34
# COMMONALITY               0.50   :   foreignness               0.33
# DIFFERENCE                0.50   :   SAMENESS                  0.33
# OTHERNESS                 0.48   :   DIFFERENCE                0.33
#
# EFFECTIVE_WAY                    :   EFFICIENT_USE                 (nn_weighted_adjective_noun_identity)
# EFFECTIVENESS             0.75   :   EFFECTIVENESS             0.48
# efficacy                  0.48   :   PRACTICALITY              0.39
# potency                   0.44   :   ACCURACY                  0.38
# PRACTICALITY              0.43   :   quality                   0.34
# ACCURACY                  0.41   :   utility                   0.34
#
# NEW_LIFE                         :   EARLY_AGE                     (nn_weighted_adjective_noun_identity)
# freshness                 0.43   :   age                       0.62
# permanence                0.43   :   timing                    0.58
# individuality             0.43   :   height                    0.33
# originality               0.42   :   maturity                  0.32
# dullness                  0.42   :   duration                  0.32
#
# LARGE_QUANTITY                   :   GREAT_MAJORITY                (nn_weighted_adjective_noun_identity)
# quantity                  0.78   :   majority                  0.70
# SIZE                      0.63   :   quality                   0.41
# width                     0.40   :   minority                  0.41
# thickness                 0.38   :   SIZE                      0.32
# volume                    0.36   :   cleanness                 0.32
#
# LONG_PERIOD                      :   SHORT_TIME                    (nn_weighted_adjective_noun_identity)
# DURATION                  0.69   :   DURATION                  0.74
# LENGTH                    0.41   :   LENGTH                    0.58
# DISTANCE                  0.36   :   DISTANCE                  0.43
# CYCLICITY                 0.35   :   CYCLICITY                 0.42
# fullness                  0.31   :   quantity                  0.41
#
# RIGHT_HAND                       :   LEFT_ARM                      (nn_weighted_adjective_noun_identity)
# POSITION                  0.62   :   POSITION                  0.48
# correctness               0.44   :   timing                    0.33
# rightness                 0.36   :   power                     0.31
# propriety                 0.34   :   length                    0.29
# direction                 0.34   :   strength                  0.25
#
# EFFECTIVE_WAY                    :   PRACTICAL_DIFFICULTY          (nn_weighted_adjective_noun_identity)
# effectiveness             0.75   :   difficulty                0.71
# efficacy                  0.48   :   PRACTICALITY              0.59
# potency                   0.44   :   complexity                0.49
# PRACTICALITY              0.43   :   necessity                 0.37
# accuracy                  0.41   :   importance                0.35
#
# MAJOR_ISSUE                      :   SOCIAL_EVENT                  (nn_weighted_adjective_noun_identity)
# significance              0.46   :   sociality                 0.46
# importance                0.44   :   sociability               0.42
# size                      0.35   :   equality                  0.35
# stature                   0.34   :   morality                  0.31
# foreignness               0.32   :   otherness                 0.30
#
# HOT_WEATHER                      :   COLD_AIR                      (nn_weighted_adjective_noun_identity)
# TEMPERATURE               0.66   :   TEMPERATURE               0.67
# WETNESS                   0.45   :   WETNESS                   0.41
# emotionality              0.40   :   accuracy                  0.37
# pleasantness              0.37   :   freshness                 0.34
# cyclicity                 0.35   :   purity                    0.34
#
# WHOLE_COUNTRY                    :   GENERAL_PRINCIPLE             (nn_weighted_adjective_noun_identity)
# integrity                 0.51   :   generality                0.58
# originality               0.44   :   concreteness              0.35
# quality                   0.44   :   equality                  0.33
# evenness                  0.43   :   humanness                 0.33
# cleanness                 0.42   :   morality                  0.33
#
# GENERAL_PRINCIPLE                :   BASIC_RULE                    (nn_weighted_adjective_noun_identity)
# GENERALITY                0.58   :   GENERALITY                0.44
# concreteness              0.35   :   humaneness                0.43
# equality                  0.33   :   practicality              0.43
# humanness                 0.33   :   comprehensiveness         0.41
# morality                  0.33   :   reasonableness            0.41
#
# DIFFERENT_KIND                   :   VARIOUS_FORM                  (nn_weighted_adjective_noun_identity)
# SAMENESS                  0.57   :   OTHERNESS                 0.39
# similarity                0.54   :   COMMONALITY               0.34
# COMMONALITY               0.50   :   foreignness               0.33
# DIFFERENCE                0.50   :   SAMENESS                  0.33
# OTHERNESS                 0.48   :   DIFFERENCE                0.33
#
# LARGE_QUANTITY                   :   GREAT_MAJORITY                (nn_weighted_adjective_noun_identity)
# quantity                  0.78   :   majority                  0.70
# SIZE                      0.63   :   quality                   0.41
# width                     0.40   :   minority                  0.41
# thickness                 0.38   :   SIZE                      0.32
# volume                    0.36   :   cleanness                 0.32
#
# SOCIAL_ACTIVITY                  :   POLITICAL_ACTION              (nn_weighted_adjective_noun_identity)
# sociability               0.45   :   action                    0.50
# sociality                 0.40   :   civility                  0.34
# otherness                 0.32   :   moderation                0.32
# domesticity               0.30   :   nastiness                 0.29
# morality                  0.30   :   fairness                  0.29
#
# PUBLIC_BUILDING                  :   CENTRAL_AUTHORITY             (nn_weighted_adjective_noun_identity)
# ordinariness              0.35   :   importance                0.39
# modesty                   0.33   :   significance              0.38
# height                    0.33   :   power                     0.32
# humanness                 0.33   :   potency                   0.31
# morality                  0.33   :   responsibility            0.30
#
# NEW_LAW                          :   BASIC_RULE                    (nn_weighted_adjective_noun_identity)
# conventionality           0.35   :   generality                0.44
# abstractness              0.34   :   humaneness                0.43
# originality               0.34   :   practicality              0.43
# liveliness                0.33   :   comprehensiveness         0.41
# individuality             0.33   :   reasonableness            0.41
#
# EFFECTIVE_WAY                    :   EFFICIENT_USE                 (nn_weighted_adjective_noun_identity)
# EFFECTIVENESS             0.75   :   EFFECTIVENESS             0.48
# efficacy                  0.48   :   PRACTICALITY              0.39
# potency                   0.44   :   ACCURACY                  0.38
# PRACTICALITY              0.43   :   quality                   0.34
# ACCURACY                  0.41   :   utility                   0.34
#
# ECONOMIC_DEVELOPMENT             :   RURAL_COMMUNITY               (nn_weighted_adjective_noun_identity)
# commerce                  0.38   :   literacy                  0.35
# completeness              0.29   :   PRACTICALITY              0.34
# auspiciousness            0.28   :   importance                0.31
# PRACTICALITY              0.28   :   ordinariness              0.31
# attractiveness            0.28   :   cleanness                 0.29
#
# SOCIAL_ACTIVITY                  :   ECONOMIC_CONDITION            (nn_weighted_adjective_noun_identity)
# sociability               0.45   :   health                    0.35
# sociality                 0.40   :   commerce                  0.33
# otherness                 0.32   :   importance                0.31
# domesticity               0.30   :   necessity                 0.31
# MORALITY                  0.30   :   MORALITY                  0.30
#
# GOOD_PLACE                       :   HIGH_POINT                    (nn_weighted_adjective_noun_identity)
# quality                   0.63   :   degree                    0.69
# cleanness                 0.48   :   height                    0.54
# consistency               0.48   :   distance                  0.29
# freshness                 0.46   :   age                       0.27
# solidity                  0.45   :   hardness                  0.27
#
# EARLY_STAGE                      :   LONG_PERIOD                   (nn_weighted_adjective_noun_identity)
# timing                    0.61   :   duration                  0.69
# height                    0.37   :   length                    0.41
# maturity                  0.30   :   distance                  0.36
# nature                    0.29   :   cyclicity                 0.35
# seriousness               0.28   :   fullness                  0.31
#
# ECONOMIC_PROBLEM                 :   PRACTICAL_DIFFICULTY          (nn_weighted_adjective_noun_identity)
# crisis                    0.40   :   difficulty                0.71
# commerce                  0.36   :   PRACTICALITY              0.59
# moderation                0.33   :   COMPLEXITY                0.49
# PRACTICALITY              0.32   :   necessity                 0.37
# COMPLEXITY                0.31   :   importance                0.35
#
# EFFECTIVE_WAY                    :   IMPORTANT_PART                (nn_weighted_adjective_noun_identity)
# effectiveness             0.75   :   importance                0.62
# efficacy                  0.48   :   significance              0.54
# potency                   0.44   :   necessity                 0.41
# practicality              0.43   :   pride                     0.39
# accuracy                  0.41   :   responsibility            0.35
#
# EARLY_EVENING                    :   PREVIOUS_DAY                  (nn_weighted_adjective_noun_identity)
# TIMING                    0.70   :   ordinariness              0.46
# duration                  0.31   :   conventionality           0.43
# accuracy                  0.31   :   sameness                  0.42
# completeness              0.31   :   TIMING                    0.42
# cyclicity                 0.29   :   similarity                0.42
#
# DIFFERENT_PART                   :   VARIOUS_FORM                  (nn_weighted_adjective_noun_identity)
# DIFFERENCE                0.56   :   otherness                 0.39
# similarity                0.51   :   COMMONALITY               0.34
# COMMONALITY               0.50   :   foreignness               0.33
# SAMENESS                  0.45   :   SAMENESS                  0.33
# individuality             0.42   :   DIFFERENCE                0.33
#
# NEW_SITUATION                    :   DIFFERENT_KIND                (nn_weighted_adjective_noun_identity)
# difficulty                0.47   :   sameness                  0.57
# complexity                0.43   :   similarity                0.54
# abstractness              0.42   :   commonality               0.50
# concreteness              0.40   :   difference                0.50
# necessity                 0.39   :   otherness                 0.48
#
# EFFICIENT_USE                    :   SIGNIFICANT_ROLE              (nn_weighted_adjective_noun_identity)
# effectiveness             0.48   :   significance              0.56
# practicality              0.39   :   importance                0.53
# accuracy                  0.38   :   status                    0.40
# quality                   0.34   :   responsibility            0.36
# utility                   0.34   :   likelihood                0.36
#
# BLACK_HAIR                       :   DARK_EYE                      (nn_weighted_adjective_noun_identity)
# evil                      0.45   :   luminosity                0.48
# length                    0.37   :   light                     0.43
# morality                  0.34   :   evenness                  0.37
# texture                   0.32   :   opacity                   0.35
# complexion                0.32   :   sharpness                 0.34
#
# AMERICAN_COUNTRY                 :   EUROPEAN_STATE                (nn_weighted_adjective_noun_identity)
# foreignness               0.32   :   potency                   0.23
# corruptness               0.27   :   solidity                  0.20
# cleanness                 0.25   :   typicality                0.20
# PURITY                    0.25   :   utility                   0.19
# niceness                  0.25   :   PURITY                    0.19
#
# SMALL_HOUSE                      :   LITTLE_ROOM                   (nn_weighted_adjective_noun_identity)
# SIZE                      0.77   :   SIZE                      0.48
# HEIGHT                    0.50   :   HEIGHT                    0.44
# width                     0.46   :   niceness                  0.42
# length                    0.39   :   cleanness                 0.42
# stature                   0.39   :   hardness                  0.41
#
# NEW_BODY                         :   WHOLE_SYSTEM                  (nn_weighted_adjective_noun_identity)
# FRESHNESS                 0.34   :   integrity                 0.58
# originality               0.33   :   quality                   0.41
# cleanness                 0.33   :   fairness                  0.39
# permanence                0.32   :   honesty                   0.39
# abstractness              0.31   :   FRESHNESS                 0.38
#
# GENERAL_PRINCIPLE                :   PRESENT_POSITION              (nn_weighted_adjective_noun_identity)
# generality                0.58   :   position                  0.61
# concreteness              0.35   :   presence                  0.38
# equality                  0.33   :   strength                  0.31
# humanness                 0.33   :   otherness                 0.31
# morality                  0.33   :   stature                   0.30
#
# EARLY_STAGE                      :   LONG_PERIOD                   (nn_weighted_adjective_noun_identity)
# timing                    0.61   :   duration                  0.69
# height                    0.37   :   length                    0.41
# maturity                  0.30   :   distance                  0.36
# nature                    0.29   :   cyclicity                 0.35
# seriousness               0.28   :   fullness                  0.31
#
# ECONOMIC_PROBLEM                 :   PRACTICAL_DIFFICULTY          (nn_weighted_adjective_noun_identity)
# crisis                    0.40   :   difficulty                0.71
# commerce                  0.36   :   PRACTICALITY              0.59
# moderation                0.33   :   COMPLEXITY                0.49
# PRACTICALITY              0.32   :   necessity                 0.37
# COMPLEXITY                0.31   :   importance                0.35
#
# EARLY_EVENING                    :   PREVIOUS_DAY                  (nn_weighted_adjective_noun_identity)
# TIMING                    0.70   :   ordinariness              0.46
# duration                  0.31   :   conventionality           0.43
# accuracy                  0.31   :   sameness                  0.42
# completeness              0.31   :   TIMING                    0.42
# cyclicity                 0.29   :   similarity                0.42
#
# NORTHERN_REGION                  :   INDUSTRIAL_AREA               (nn_weighted_adjective_noun_identity)
# distance                  0.29   :   cleanness                 0.37
# evenness                  0.27   :   naturalness               0.32
# cyclicity                 0.26   :   tractability              0.31
# otherness                 0.25   :   commerce                  0.31
# attractiveness            0.24   :   pleasantness              0.30
#
# CERTAIN_CIRCUMSTANCE             :   ECONOMIC_CONDITION            (nn_weighted_adjective_noun_identity)
# commonness                0.42   :   health                    0.35
# otherness                 0.40   :   commerce                  0.33
# foreignness               0.39   :   importance                0.31
# concreteness              0.37   :   necessity                 0.31
# ordinariness              0.37   :   morality                  0.30
#
# HIGH_PRICE                       :   LOW_COST                      (nn_weighted_adjective_noun_identity)
# DEGREE                    0.54   :   HEIGHT                    0.42
# HEIGHT                    0.48   :   convenience               0.35
# price                     0.46   :   DEGREE                    0.34
# moderation                0.44   :   ease                      0.32
# quantity                  0.33   :   dispensability            0.32
#
# DARK_EYE                         :   LEFT_ARM                      (nn_weighted_adjective_noun_identity)
# luminosity                0.48   :   position                  0.48
# light                     0.43   :   timing                    0.33
# evenness                  0.37   :   power                     0.31
# opacity                   0.35   :   length                    0.29
# sharpness                 0.34   :   strength                  0.25
#
# NEW_LIFE                         :   EARLY_AGE                     (nn_weighted_adjective_noun_identity)
# freshness                 0.43   :   age                       0.62
# permanence                0.43   :   timing                    0.58
# individuality             0.43   :   height                    0.33
# originality               0.42   :   maturity                  0.32
# dullness                  0.42   :   duration                  0.32
#
# OLDER_MAN                        :   ELDERLY_WOMAN                 (nn_weighted_adjective_noun_identity)
# AGE                       0.56   :   AGE                       0.39
# playfulness               0.40   :   fertility                 0.34
# emotionality              0.38   :   health                    0.32
# sociability               0.38   :   COMMONNESS                0.32
# COMMONNESS                0.38   :   equality                  0.30
#
# NEW_INFORMATION                  :   FURTHER_EVIDENCE              (nn_weighted_adjective_noun_identity)
# individuality             0.46   :   possibility               0.36
# originality               0.44   :   finality                  0.36
# permanence                0.44   :   credibility               0.36
# immediacy                 0.44   :   sharpness                 0.34
# correctness               0.42   :   freshness                 0.34
#
# EFFECTIVE_WAY                    :   IMPORTANT_PART                (nn_weighted_adjective_noun_identity)
# effectiveness             0.75   :   importance                0.62
# efficacy                  0.48   :   significance              0.54
# potency                   0.44   :   necessity                 0.41
# practicality              0.43   :   pride                     0.39
# accuracy                  0.41   :   responsibility            0.35
#
# BETTER_JOB                       :   GOOD_EFFECT                   (nn_weighted_adjective_noun_identity)
# ease                      0.41   :   QUALITY                   0.50
# QUALITY                   0.40   :   effectiveness             0.43
# difficulty                0.40   :   friendliness              0.37
# naturalness               0.40   :   worthiness                0.37
# perfection                0.39   :   freshness                 0.37
#
# GOOD_PLACE                       :   HIGH_POINT                    (nn_weighted_adjective_noun_identity)
# quality                   0.63   :   degree                    0.69
# cleanness                 0.48   :   height                    0.54
# consistency               0.48   :   distance                  0.29
# freshness                 0.46   :   age                       0.27
# solidity                  0.45   :   hardness                  0.27
#
# GENERAL_PRINCIPLE                :   PRESENT_POSITION              (nn_weighted_adjective_noun_identity)
# generality                0.58   :   position                  0.61
# concreteness              0.35   :   presence                  0.38
# equality                  0.33   :   strength                  0.31
# humanness                 0.33   :   otherness                 0.31
# morality                  0.33   :   stature                   0.30
#
# DIFFERENT_PART                   :   VARIOUS_FORM                  (nn_weighted_adjective_noun_identity)
# DIFFERENCE                0.56   :   otherness                 0.39
# similarity                0.51   :   COMMONALITY               0.34
# COMMONALITY               0.50   :   foreignness               0.33
# SAMENESS                  0.45   :   SAMENESS                  0.33
# individuality             0.42   :   DIFFERENCE                0.33
#
# NEW_SITUATION                    :   DIFFERENT_KIND                (nn_weighted_adjective_noun_identity)
# difficulty                0.47   :   sameness                  0.57
# complexity                0.43   :   similarity                0.54
# abstractness              0.42   :   commonality               0.50
# concreteness              0.40   :   difference                0.50
# necessity                 0.39   :   otherness                 0.48
#
# GOOD_PLACE                       :   HIGH_POINT                    (nn_weighted_adjective_noun_identity)
# quality                   0.63   :   degree                    0.69
# cleanness                 0.48   :   height                    0.54
# consistency               0.48   :   distance                  0.29
# freshness                 0.46   :   age                       0.27
# solidity                  0.45   :   hardness                  0.27
#
# NEW_INFORMATION                  :   FURTHER_EVIDENCE              (nn_weighted_adjective_noun_identity)
# individuality             0.46   :   possibility               0.36
# originality               0.44   :   finality                  0.36
# permanence                0.44   :   credibility               0.36
# immediacy                 0.44   :   sharpness                 0.34
# correctness               0.42   :   freshness                 0.34
#
# SPECIAL_CIRCUMSTANCE             :   PARTICULAR_CASE               (nn_weighted_adjective_noun_identity)
# ordinariness              0.47   :   significance              0.49
# COMMONNESS                0.44   :   importance                0.45
# otherness                 0.40   :   COMMONNESS                0.43
# nobility                  0.37   :   abstractness              0.41
# gregariousness            0.37   :   seriousness               0.40
#
# NEW_SITUATION                    :   DIFFERENT_KIND                (nn_weighted_adjective_noun_identity)
# difficulty                0.47   :   sameness                  0.57
# complexity                0.43   :   similarity                0.54
# abstractness              0.42   :   commonality               0.50
# concreteness              0.40   :   difference                0.50
# necessity                 0.39   :   otherness                 0.48
#
# VAST_AMOUNT                      :   LARGE_QUANTITY                (nn_weighted_adjective_noun_identity)
# QUANTITY                  0.58   :   QUANTITY                  0.78
# SIZE                      0.57   :   SIZE                      0.63
# WIDTH                     0.42   :   WIDTH                     0.40
# THICKNESS                 0.38   :   THICKNESS                 0.38
# substantiality            0.35   :   volume                    0.36
#
# POLITICAL_ACTION                 :   ECONOMIC_DEVELOPMENT          (nn_weighted_adjective_noun_identity)
# action                    0.50   :   commerce                  0.38
# civility                  0.34   :   completeness              0.29
# moderation                0.32   :   auspiciousness            0.28
# nastiness                 0.29   :   practicality              0.28
# fairness                  0.29   :   attractiveness            0.28
#
# NEW_LIFE                         :   EARLY_AGE                     (nn_weighted_adjective_noun_identity)
# freshness                 0.43   :   age                       0.62
# permanence                0.43   :   timing                    0.58
# individuality             0.43   :   height                    0.33
# originality               0.42   :   maturity                  0.32
# dullness                  0.42   :   duration                  0.32
#
# BETTER_JOB                       :   GOOD_PLACE                    (nn_weighted_adjective_noun_identity)
# ease                      0.41   :   QUALITY                   0.63
# QUALITY                   0.40   :   cleanness                 0.48
# difficulty                0.40   :   consistency               0.48
# naturalness               0.40   :   freshness                 0.46
# perfection                0.39   :   solidity                  0.45
#
# VAST_AMOUNT                      :   HIGH_PRICE                    (nn_weighted_adjective_noun_identity)
# QUANTITY                  0.58   :   degree                    0.54
# size                      0.57   :   height                    0.48
# width                     0.42   :   price                     0.46
# thickness                 0.38   :   moderation                0.44
# substantiality            0.35   :   QUANTITY                  0.33
#
# RIGHT_HAND                       :   LEFT_ARM                      (nn_weighted_adjective_noun_identity)
# POSITION                  0.62   :   POSITION                  0.48
# correctness               0.44   :   timing                    0.33
# rightness                 0.36   :   power                     0.31
# propriety                 0.34   :   length                    0.29
# direction                 0.34   :   strength                  0.25
#
# NEW_LAW                          :   BASIC_RULE                    (nn_weighted_adjective_noun_identity)
# conventionality           0.35   :   generality                0.44
# abstractness              0.34   :   humaneness                0.43
# originality               0.34   :   practicality              0.43
# liveliness                0.33   :   comprehensiveness         0.41
# individuality             0.33   :   reasonableness            0.41
#
# DIFFERENT_PART                   :   NORTHERN_REGION               (nn_weighted_adjective_noun_identity)
# difference                0.56   :   distance                  0.29
# similarity                0.51   :   evenness                  0.27
# commonality               0.50   :   cyclicity                 0.26
# sameness                  0.45   :   otherness                 0.25
# individuality             0.42   :   attractiveness            0.24
#
# SIMILAR_RESULT                   :   GOOD_EFFECT                   (nn_weighted_adjective_noun_identity)
# similarity                0.62   :   quality                   0.50
# commonality               0.45   :   effectiveness             0.43
# difference                0.40   :   friendliness              0.37
# familiarity               0.37   :   worthiness                0.37
# connection                0.36   :   freshness                 0.37
#
# LARGE_NUMBER                     :   GREAT_MAJORITY                (nn_weighted_adjective_noun_identity)
# SIZE                      0.67   :   majority                  0.70
# quantity                  0.45   :   quality                   0.41
# width                     0.38   :   minority                  0.41
# volume                    0.38   :   SIZE                      0.32
# complexity                0.34   :   cleanness                 0.32
#
# CERTAIN_CIRCUMSTANCE             :   PARTICULAR_CASE               (nn_weighted_adjective_noun_identity)
# COMMONNESS                0.42   :   significance              0.49
# otherness                 0.40   :   importance                0.45
# foreignness               0.39   :   COMMONNESS                0.43
# concreteness              0.37   :   abstractness              0.41
# ordinariness              0.37   :   seriousness               0.40
#
# LONG_PERIOD                      :   SHORT_TIME                    (nn_weighted_adjective_noun_identity)
# DURATION                  0.69   :   DURATION                  0.74
# LENGTH                    0.41   :   LENGTH                    0.58
# DISTANCE                  0.36   :   DISTANCE                  0.43
# CYCLICITY                 0.35   :   CYCLICITY                 0.42
# fullness                  0.31   :   quantity                  0.41
#
# NEW_LAW                          :   BASIC_RULE                    (nn_weighted_adjective_noun_identity)
# conventionality           0.35   :   generality                0.44
# abstractness              0.34   :   humaneness                0.43
# originality               0.34   :   practicality              0.43
# liveliness                0.33   :   comprehensiveness         0.41
# individuality             0.33   :   reasonableness            0.41
#
# PREVIOUS_DAY                     :   EARLY_AGE                     (nn_weighted_adjective_noun_identity)
# ordinariness              0.46   :   age                       0.62
# conventionality           0.43   :   TIMING                    0.58
# sameness                  0.42   :   height                    0.33
# TIMING                    0.42   :   maturity                  0.32
# similarity                0.42   :   duration                  0.32
#
# HOT_WEATHER                      :   COLD_AIR                      (nn_weighted_adjective_noun_identity)
# TEMPERATURE               0.66   :   TEMPERATURE               0.67
# WETNESS                   0.45   :   WETNESS                   0.41
# emotionality              0.40   :   accuracy                  0.37
# pleasantness              0.37   :   freshness                 0.34
# cyclicity                 0.35   :   purity                    0.34
#
# MAJOR_ISSUE                      :   SOCIAL_EVENT                  (nn_weighted_adjective_noun_identity)
# significance              0.46   :   sociality                 0.46
# importance                0.44   :   sociability               0.42
# size                      0.35   :   equality                  0.35
# stature                   0.34   :   morality                  0.31
# foreignness               0.32   :   otherness                 0.30
#
# DIFFERENT_PART                   :   NORTHERN_REGION               (nn_weighted_adjective_noun_identity)
# difference                0.56   :   distance                  0.29
# similarity                0.51   :   evenness                  0.27
# commonality               0.50   :   cyclicity                 0.26
# sameness                  0.45   :   otherness                 0.25
# individuality             0.42   :   attractiveness            0.24
#
# HIGH_PRICE                       :   LOW_COST                      (nn_weighted_adjective_noun_identity)
# DEGREE                    0.54   :   HEIGHT                    0.42
# HEIGHT                    0.48   :   convenience               0.35
# price                     0.46   :   DEGREE                    0.34
# moderation                0.44   :   ease                      0.32
# quantity                  0.33   :   dispensability            0.32
#
# EARLIER_WORK                     :   EARLY_STAGE                   (nn_weighted_adjective_noun_identity)
# TIMING                    0.34   :   TIMING                    0.61
# regularity                0.33   :   height                    0.37
# accuracy                  0.33   :   maturity                  0.30
# otherness                 0.33   :   nature                    0.29
# ordinariness              0.32   :   seriousness               0.28
#
# OLDER_MAN                        :   ELDERLY_WOMAN                 (nn_weighted_adjective_noun_identity)
# AGE                       0.56   :   AGE                       0.39
# playfulness               0.40   :   fertility                 0.34
# emotionality              0.38   :   health                    0.32
# sociability               0.38   :   COMMONNESS                0.32
# COMMONNESS                0.38   :   equality                  0.30
#
# FEDERAL_ASSEMBLY                 :   NATIONAL_GOVERNMENT           (nn_weighted_adjective_noun_identity)
# majority                  0.23   :   importance                0.35
# otherness                 0.22   :   significance              0.31
# nature                    0.22   :   stature                   0.30
# seniority                 0.21   :   seriousness               0.29
# sameness                  0.21   :   intrusiveness             0.29
#
# OLD_PERSON                       :   ELDERLY_LADY                  (nn_weighted_adjective_noun_identity)
# AGE                       0.45   :   commonness                0.31
# otherness                 0.40   :   AGE                       0.30
# typicality                0.40   :   health                    0.29
# abstractness              0.37   :   thoughtfulness            0.29
# originality               0.36   :   kindness                  0.29
#
# GENERAL_PRINCIPLE                :   BASIC_RULE                    (nn_weighted_adjective_noun_identity)
# GENERALITY                0.58   :   GENERALITY                0.44
# concreteness              0.35   :   humaneness                0.43
# equality                  0.33   :   practicality              0.43
# humanness                 0.33   :   comprehensiveness         0.41
# morality                  0.33   :   reasonableness            0.41
#
# LARGE_QUANTITY                   :   GREAT_MAJORITY                (nn_weighted_adjective_noun_identity)
# quantity                  0.78   :   majority                  0.70
# SIZE                      0.63   :   quality                   0.41
# width                     0.40   :   minority                  0.41
# thickness                 0.38   :   SIZE                      0.32
# volume                    0.36   :   cleanness                 0.32
#
# EFFECTIVE_WAY                    :   EFFICIENT_USE                 (nn_weighted_adjective_noun_identity)
# EFFECTIVENESS             0.75   :   EFFECTIVENESS             0.48
# efficacy                  0.48   :   PRACTICALITY              0.39
# potency                   0.44   :   ACCURACY                  0.38
# PRACTICALITY              0.43   :   quality                   0.34
# ACCURACY                  0.41   :   utility                   0.34
#
# FEDERAL_ASSEMBLY                 :   NATIONAL_GOVERNMENT           (nn_weighted_adjective_noun_identity)
# majority                  0.23   :   importance                0.35
# otherness                 0.22   :   significance              0.31
# nature                    0.22   :   stature                   0.30
# seniority                 0.21   :   seriousness               0.29
# sameness                  0.21   :   intrusiveness             0.29
#
# SPECIAL_CIRCUMSTANCE             :   PARTICULAR_CASE               (nn_weighted_adjective_noun_identity)
# ordinariness              0.47   :   significance              0.49
# COMMONNESS                0.44   :   importance                0.45
# otherness                 0.40   :   COMMONNESS                0.43
# nobility                  0.37   :   abstractness              0.41
# gregariousness            0.37   :   seriousness               0.40
#
# SMALL_HOUSE                      :   LITTLE_ROOM                   (nn_weighted_adjective_noun_identity)
# SIZE                      0.77   :   SIZE                      0.48
# HEIGHT                    0.50   :   HEIGHT                    0.44
# width                     0.46   :   niceness                  0.42
# length                    0.39   :   cleanness                 0.42
# stature                   0.39   :   hardness                  0.41
#
# NEW_INFORMATION                  :   FURTHER_EVIDENCE              (nn_weighted_adjective_noun_identity)
# individuality             0.46   :   possibility               0.36
# originality               0.44   :   finality                  0.36
# permanence                0.44   :   credibility               0.36
# immediacy                 0.44   :   sharpness                 0.34
# correctness               0.42   :   freshness                 0.34
#
# SPECIAL_CIRCUMSTANCE             :   PARTICULAR_CASE               (nn_weighted_adjective_noun_identity)
# ordinariness              0.47   :   significance              0.49
# COMMONNESS                0.44   :   importance                0.45
# otherness                 0.40   :   COMMONNESS                0.43
# nobility                  0.37   :   abstractness              0.41
# gregariousness            0.37   :   seriousness               0.40
#
# ECONOMIC_PROBLEM                 :   PRACTICAL_DIFFICULTY          (nn_weighted_adjective_noun_identity)
# crisis                    0.40   :   difficulty                0.71
# commerce                  0.36   :   PRACTICALITY              0.59
# moderation                0.33   :   COMPLEXITY                0.49
# PRACTICALITY              0.32   :   necessity                 0.37
# COMPLEXITY                0.31   :   importance                0.35
#
# EFFECTIVE_WAY                    :   IMPORTANT_PART                (nn_weighted_adjective_noun_identity)
# effectiveness             0.75   :   importance                0.62
# efficacy                  0.48   :   significance              0.54
# potency                   0.44   :   necessity                 0.41
# practicality              0.43   :   pride                     0.39
# accuracy                  0.41   :   responsibility            0.35
#
# HIGH_PRICE                       :   SHORT_TIME                    (nn_weighted_adjective_noun_identity)
# degree                    0.54   :   duration                  0.74
# height                    0.48   :   length                    0.58
# price                     0.46   :   distance                  0.43
# moderation                0.44   :   cyclicity                 0.42
# QUANTITY                  0.33   :   QUANTITY                  0.41
#
# NEW_SITUATION                    :   DIFFERENT_KIND                (nn_weighted_adjective_noun_identity)
# difficulty                0.47   :   sameness                  0.57
# complexity                0.43   :   similarity                0.54
# abstractness              0.42   :   commonality               0.50
# concreteness              0.40   :   difference                0.50
# necessity                 0.39   :   otherness                 0.48
#
# GOOD_PLACE                       :   HIGH_POINT                    (nn_weighted_adjective_noun_identity)
# quality                   0.63   :   degree                    0.69
# cleanness                 0.48   :   height                    0.54
# consistency               0.48   :   distance                  0.29
# freshness                 0.46   :   age                       0.27
# solidity                  0.45   :   hardness                  0.27
#
# NEW_SITUATION                    :   DIFFERENT_KIND                (nn_weighted_adjective_noun_identity)
# difficulty                0.47   :   sameness                  0.57
# complexity                0.43   :   similarity                0.54
# abstractness              0.42   :   commonality               0.50
# concreteness              0.40   :   difference                0.50
# necessity                 0.39   :   otherness                 0.48
#
# CERTAIN_CIRCUMSTANCE             :   PARTICULAR_CASE               (nn_weighted_adjective_noun_identity)
# COMMONNESS                0.42   :   significance              0.49
# otherness                 0.40   :   importance                0.45
# foreignness               0.39   :   COMMONNESS                0.43
# concreteness              0.37   :   abstractness              0.41
# ordinariness              0.37   :   seriousness               0.40
#
# NEW_LAW                          :   BASIC_RULE                    (nn_weighted_adjective_noun_identity)
# conventionality           0.35   :   generality                0.44
# abstractness              0.34   :   humaneness                0.43
# originality               0.34   :   practicality              0.43
# liveliness                0.33   :   comprehensiveness         0.41
# individuality             0.33   :   reasonableness            0.41
#
# EFFECTIVE_WAY                    :   EFFICIENT_USE                 (nn_weighted_adjective_noun_identity)
# EFFECTIVENESS             0.75   :   EFFECTIVENESS             0.48
# efficacy                  0.48   :   PRACTICALITY              0.39
# potency                   0.44   :   ACCURACY                  0.38
# PRACTICALITY              0.43   :   quality                   0.34
# ACCURACY                  0.41   :   utility                   0.34
#
# OLDER_MAN                        :   ELDERLY_WOMAN                 (nn_weighted_adjective_noun_identity)
# AGE                       0.56   :   AGE                       0.39
# playfulness               0.40   :   fertility                 0.34
# emotionality              0.38   :   health                    0.32
# sociability               0.38   :   COMMONNESS                0.32
# COMMONNESS                0.38   :   equality                  0.30
#
# FEDERAL_ASSEMBLY                 :   NATIONAL_GOVERNMENT           (nn_weighted_adjective_noun_identity)
# majority                  0.23   :   importance                0.35
# otherness                 0.22   :   significance              0.31
# nature                    0.22   :   stature                   0.30
# seniority                 0.21   :   seriousness               0.29
# sameness                  0.21   :   intrusiveness             0.29
#
# SPECIAL_CIRCUMSTANCE             :   ELDERLY_LADY                  (nn_weighted_adjective_noun_identity)
# ordinariness              0.47   :   COMMONNESS                0.31
# COMMONNESS                0.44   :   age                       0.30
# otherness                 0.40   :   health                    0.29
# nobility                  0.37   :   thoughtfulness            0.29
# gregariousness            0.37   :   kindness                  0.29
#
# CERTAIN_CIRCUMSTANCE             :   PARTICULAR_CASE               (nn_weighted_adjective_noun_identity)
# COMMONNESS                0.42   :   significance              0.49
# otherness                 0.40   :   importance                0.45
# foreignness               0.39   :   COMMONNESS                0.43
# concreteness              0.37   :   abstractness              0.41
# ordinariness              0.37   :   seriousness               0.40
#
# SOCIAL_ACTIVITY                  :   POLITICAL_ACTION              (nn_weighted_adjective_noun_identity)
# sociability               0.45   :   action                    0.50
# sociality                 0.40   :   civility                  0.34
# otherness                 0.32   :   moderation                0.32
# domesticity               0.30   :   nastiness                 0.29
# morality                  0.30   :   fairness                  0.29
#
# IMPORTANT_PART                   :   SIGNIFICANT_ROLE              (nn_weighted_adjective_noun_identity)
# IMPORTANCE                0.62   :   SIGNIFICANCE              0.56
# SIGNIFICANCE              0.54   :   IMPORTANCE                0.53
# necessity                 0.41   :   status                    0.40
# pride                     0.39   :   RESPONSIBILITY            0.36
# RESPONSIBILITY            0.35   :   likelihood                0.36
#
# CENTRAL_AUTHORITY                :   LOCAL_OFFICE                  (nn_weighted_adjective_noun_identity)
# importance                0.39   :   handiness                 0.33
# significance              0.38   :   cleanness                 0.30
# power                     0.32   :   immediacy                 0.30
# potency                   0.31   :   freshness                 0.30
# responsibility            0.30   :   friendliness              0.29
#
# EFFECTIVE_WAY                    :   EFFICIENT_USE                 (nn_weighted_adjective_noun_identity)
# EFFECTIVENESS             0.75   :   EFFECTIVENESS             0.48
# efficacy                  0.48   :   PRACTICALITY              0.39
# potency                   0.44   :   ACCURACY                  0.38
# PRACTICALITY              0.43   :   quality                   0.34
# ACCURACY                  0.41   :   utility                   0.34
#
# SOCIAL_EVENT                     :   SPECIAL_CIRCUMSTANCE          (nn_weighted_adjective_noun_identity)
# sociality                 0.46   :   ordinariness              0.47
# sociability               0.42   :   commonness                0.44
# equality                  0.35   :   OTHERNESS                 0.40
# morality                  0.31   :   nobility                  0.37
# OTHERNESS                 0.30   :   gregariousness            0.37
#
# BETTER_JOB                       :   GOOD_PLACE                    (nn_weighted_adjective_noun_identity)
# ease                      0.41   :   QUALITY                   0.63
# QUALITY                   0.40   :   cleanness                 0.48
# difficulty                0.40   :   consistency               0.48
# naturalness               0.40   :   freshness                 0.46
# perfection                0.39   :   solidity                  0.45
#
# FEDERAL_ASSEMBLY                 :   NATIONAL_GOVERNMENT           (nn_weighted_adjective_noun_identity)
# majority                  0.23   :   importance                0.35
# otherness                 0.22   :   significance              0.31
# nature                    0.22   :   stature                   0.30
# seniority                 0.21   :   seriousness               0.29
# sameness                  0.21   :   intrusiveness             0.29
#
# LARGE_QUANTITY                   :   GREAT_MAJORITY                (nn_weighted_adjective_noun_identity)
# quantity                  0.78   :   majority                  0.70
# SIZE                      0.63   :   quality                   0.41
# width                     0.40   :   minority                  0.41
# thickness                 0.38   :   SIZE                      0.32
# volume                    0.36   :   cleanness                 0.32
#
# SIMILAR_RESULT                   :   GOOD_EFFECT                   (nn_weighted_adjective_noun_identity)
# similarity                0.62   :   quality                   0.50
# commonality               0.45   :   effectiveness             0.43
# difference                0.40   :   friendliness              0.37
# familiarity               0.37   :   worthiness                0.37
# connection                0.36   :   freshness                 0.37
#
# NEW_LAW                          :   BASIC_RULE                    (nn_weighted_adjective_noun_identity)
# conventionality           0.35   :   generality                0.44
# abstractness              0.34   :   humaneness                0.43
# originality               0.34   :   practicality              0.43
# liveliness                0.33   :   comprehensiveness         0.41
# individuality             0.33   :   reasonableness            0.41
#
# PUBLIC_BUILDING                  :   CENTRAL_AUTHORITY             (nn_weighted_adjective_noun_identity)
# ordinariness              0.35   :   importance                0.39
# modesty                   0.33   :   significance              0.38
# height                    0.33   :   power                     0.32
# humanness                 0.33   :   potency                   0.31
# morality                  0.33   :   responsibility            0.30
#
# MAJOR_ISSUE                      :   SOCIAL_EVENT                  (nn_weighted_adjective_noun_identity)
# significance              0.46   :   sociality                 0.46
# importance                0.44   :   sociability               0.42
# size                      0.35   :   equality                  0.35
# stature                   0.34   :   morality                  0.31
# foreignness               0.32   :   otherness                 0.30
#
# LARGE_NUMBER                     :   GREAT_MAJORITY                (nn_weighted_adjective_noun_identity)
# SIZE                      0.67   :   majority                  0.70
# quantity                  0.45   :   quality                   0.41
# width                     0.38   :   minority                  0.41
# volume                    0.38   :   SIZE                      0.32
# complexity                0.34   :   cleanness                 0.32
#
# SOCIAL_ACTIVITY                  :   POLITICAL_ACTION              (nn_weighted_adjective_noun_identity)
# sociability               0.45   :   action                    0.50
# sociality                 0.40   :   civility                  0.34
# otherness                 0.32   :   moderation                0.32
# domesticity               0.30   :   nastiness                 0.29
# morality                  0.30   :   fairness                  0.29
#
# RIGHT_HAND                       :   LEFT_ARM                      (nn_weighted_adjective_noun_identity)
# POSITION                  0.62   :   POSITION                  0.48
# correctness               0.44   :   timing                    0.33
# rightness                 0.36   :   power                     0.31
# propriety                 0.34   :   length                    0.29
# direction                 0.34   :   strength                  0.25
#
# IMPORTANT_PART                   :   SIGNIFICANT_ROLE              (nn_weighted_adjective_noun_identity)
# IMPORTANCE                0.62   :   SIGNIFICANCE              0.56
# SIGNIFICANCE              0.54   :   IMPORTANCE                0.53
# necessity                 0.41   :   status                    0.40
# pride                     0.39   :   RESPONSIBILITY            0.36
# RESPONSIBILITY            0.35   :   likelihood                0.36
#
# NEW_SITUATION                    :   PRESENT_POSITION              (nn_weighted_adjective_noun_identity)
# difficulty                0.47   :   position                  0.61
# complexity                0.43   :   presence                  0.38
# abstractness              0.42   :   strength                  0.31
# concreteness              0.40   :   otherness                 0.31
# necessity                 0.39   :   stature                   0.30
#
# OLD_PERSON                       :   ELDERLY_LADY                  (nn_weighted_adjective_noun_identity)
# AGE                       0.45   :   commonness                0.31
# otherness                 0.40   :   AGE                       0.30
# typicality                0.40   :   health                    0.29
# abstractness              0.37   :   thoughtfulness            0.29
# originality               0.36   :   kindness                  0.29
#
# NEW_LIFE                         :   EARLY_AGE                     (nn_weighted_adjective_noun_identity)
# freshness                 0.43   :   age                       0.62
# permanence                0.43   :   timing                    0.58
# individuality             0.43   :   height                    0.33
# originality               0.42   :   maturity                  0.32
# dullness                  0.42   :   duration                  0.32
#
# OLDER_MAN                        :   ELDERLY_WOMAN                 (nn_weighted_adjective_noun_identity)
# AGE                       0.56   :   AGE                       0.39
# playfulness               0.40   :   fertility                 0.34
# emotionality              0.38   :   health                    0.32
# sociability               0.38   :   COMMONNESS                0.32
# COMMONNESS                0.38   :   equality                  0.30
#
# EARLIER_WORK                     :   EARLY_STAGE                   (nn_weighted_adjective_noun_identity)
# TIMING                    0.34   :   TIMING                    0.61
# regularity                0.33   :   height                    0.37
# accuracy                  0.33   :   maturity                  0.30
# otherness                 0.33   :   nature                    0.29
# ordinariness              0.32   :   seriousness               0.28
#
# GOOD_PLACE                       :   HIGH_POINT                    (nn_weighted_adjective_noun_identity)
# quality                   0.63   :   degree                    0.69
# cleanness                 0.48   :   height                    0.54
# consistency               0.48   :   distance                  0.29
# freshness                 0.46   :   age                       0.27
# solidity                  0.45   :   hardness                  0.27
#
# ECONOMIC_DEVELOPMENT             :   RURAL_COMMUNITY               (nn_weighted_adjective_noun_identity)
# commerce                  0.38   :   literacy                  0.35
# completeness              0.29   :   PRACTICALITY              0.34
# auspiciousness            0.28   :   importance                0.31
# PRACTICALITY              0.28   :   ordinariness              0.31
# attractiveness            0.28   :   cleanness                 0.29
#
# SMALL_HOUSE                      :   LITTLE_ROOM                   (nn_weighted_adjective_noun_identity)
# SIZE                      0.77   :   SIZE                      0.48
# HEIGHT                    0.50   :   HEIGHT                    0.44
# width                     0.46   :   niceness                  0.42
# length                    0.39   :   cleanness                 0.42
# stature                   0.39   :   hardness                  0.41
#
# SIMILAR_RESULT                   :   BASIC_RULE                    (nn_weighted_adjective_noun_identity)
# similarity                0.62   :   generality                0.44
# commonality               0.45   :   humaneness                0.43
# difference                0.40   :   practicality              0.43
# familiarity               0.37   :   comprehensiveness         0.41
# connection                0.36   :   reasonableness            0.41
#
# ECONOMIC_PROBLEM                 :   PRACTICAL_DIFFICULTY          (nn_weighted_adjective_noun_identity)
# crisis                    0.40   :   difficulty                0.71
# commerce                  0.36   :   PRACTICALITY              0.59
# moderation                0.33   :   COMPLEXITY                0.49
# PRACTICALITY              0.32   :   necessity                 0.37
# COMPLEXITY                0.31   :   importance                0.35
#
# EFFECTIVE_WAY                    :   IMPORTANT_PART                (nn_weighted_adjective_noun_identity)
# effectiveness             0.75   :   importance                0.62
# efficacy                  0.48   :   significance              0.54
# potency                   0.44   :   necessity                 0.41
# practicality              0.43   :   pride                     0.39
# accuracy                  0.41   :   responsibility            0.35
#
# DIFFERENT_PART                   :   VARIOUS_FORM                  (nn_weighted_adjective_noun_identity)
# DIFFERENCE                0.56   :   otherness                 0.39
# similarity                0.51   :   COMMONALITY               0.34
# COMMONALITY               0.50   :   foreignness               0.33
# SAMENESS                  0.45   :   SAMENESS                  0.33
# individuality             0.42   :   DIFFERENCE                0.33
#
# AMERICAN_COUNTRY                 :   EUROPEAN_STATE                (nn_weighted_adjective_noun_identity)
# foreignness               0.32   :   potency                   0.23
# corruptness               0.27   :   solidity                  0.20
# cleanness                 0.25   :   typicality                0.20
# PURITY                    0.25   :   utility                   0.19
# niceness                  0.25   :   PURITY                    0.19
#
# SMALL_HOUSE                      :   LITTLE_ROOM                   (nn_weighted_adjective_noun_identity)
# SIZE                      0.77   :   SIZE                      0.48
# HEIGHT                    0.50   :   HEIGHT                    0.44
# width                     0.46   :   niceness                  0.42
# length                    0.39   :   cleanness                 0.42
# stature                   0.39   :   hardness                  0.41
#
# EARLY_STAGE                      :   LONG_PERIOD                   (nn_weighted_adjective_noun_identity)
# timing                    0.61   :   duration                  0.69
# height                    0.37   :   length                    0.41
# maturity                  0.30   :   distance                  0.36
# nature                    0.29   :   cyclicity                 0.35
# seriousness               0.28   :   fullness                  0.31
#
# EFFECTIVE_WAY                    :   IMPORTANT_PART                (nn_weighted_adjective_noun_identity)
# effectiveness             0.75   :   importance                0.62
# efficacy                  0.48   :   significance              0.54
# potency                   0.44   :   necessity                 0.41
# practicality              0.43   :   pride                     0.39
# accuracy                  0.41   :   responsibility            0.35
#
# SPECIAL_CIRCUMSTANCE             :   PARTICULAR_CASE               (nn_weighted_adjective_noun_identity)
# ordinariness              0.47   :   significance              0.49
# COMMONNESS                0.44   :   importance                0.45
# otherness                 0.40   :   COMMONNESS                0.43
# nobility                  0.37   :   abstractness              0.41
# gregariousness            0.37   :   seriousness               0.40
#
# NEW_INFORMATION                  :   FURTHER_EVIDENCE              (nn_weighted_adjective_noun_identity)
# individuality             0.46   :   possibility               0.36
# originality               0.44   :   finality                  0.36
# permanence                0.44   :   credibility               0.36
# immediacy                 0.44   :   sharpness                 0.34
# correctness               0.42   :   freshness                 0.34
#
# SMALL_HOUSE                      :   LITTLE_ROOM                   (nn_weighted_adjective_noun_identity)
# SIZE                      0.77   :   SIZE                      0.48
# HEIGHT                    0.50   :   HEIGHT                    0.44
# width                     0.46   :   niceness                  0.42
# length                    0.39   :   cleanness                 0.42
# stature                   0.39   :   hardness                  0.41
#
# SOCIAL_ACTIVITY                  :   ECONOMIC_CONDITION            (nn_weighted_adjective_noun_identity)
# sociability               0.45   :   health                    0.35
# sociality                 0.40   :   commerce                  0.33
# otherness                 0.32   :   importance                0.31
# domesticity               0.30   :   necessity                 0.31
# MORALITY                  0.30   :   MORALITY                  0.30
#
# EFFECTIVE_WAY                    :   IMPORTANT_PART                (nn_weighted_adjective_noun_identity)
# effectiveness             0.75   :   importance                0.62
# efficacy                  0.48   :   significance              0.54
# potency                   0.44   :   necessity                 0.41
# practicality              0.43   :   pride                     0.39
# accuracy                  0.41   :   responsibility            0.35
#
# DIFFERENT_PART                   :   VARIOUS_FORM                  (nn_weighted_adjective_noun_identity)
# DIFFERENCE                0.56   :   otherness                 0.39
# similarity                0.51   :   COMMONALITY               0.34
# COMMONALITY               0.50   :   foreignness               0.33
# SAMENESS                  0.45   :   SAMENESS                  0.33
# individuality             0.42   :   DIFFERENCE                0.33
#
# LARGE_NUMBER                     :   GREAT_MAJORITY                (nn_weighted_adjective_noun_identity)
# SIZE                      0.67   :   majority                  0.70
# quantity                  0.45   :   quality                   0.41
# width                     0.38   :   minority                  0.41
# volume                    0.38   :   SIZE                      0.32
# complexity                0.34   :   cleanness                 0.32
#
# SOCIAL_ACTIVITY                  :   POLITICAL_ACTION              (nn_weighted_adjective_noun_identity)
# sociability               0.45   :   action                    0.50
# sociality                 0.40   :   civility                  0.34
# otherness                 0.32   :   moderation                0.32
# domesticity               0.30   :   nastiness                 0.29
# morality                  0.30   :   fairness                  0.29
#
# WHOLE_COUNTRY                    :   GENERAL_PRINCIPLE             (nn_weighted_adjective_noun_identity)
# integrity                 0.51   :   generality                0.58
# originality               0.44   :   concreteness              0.35
# quality                   0.44   :   equality                  0.33
# evenness                  0.43   :   humanness                 0.33
# cleanness                 0.42   :   morality                  0.33
#
# CERTAIN_CIRCUMSTANCE             :   ECONOMIC_CONDITION            (nn_weighted_adjective_noun_identity)
# commonness                0.42   :   health                    0.35
# otherness                 0.40   :   commerce                  0.33
# foreignness               0.39   :   importance                0.31
# concreteness              0.37   :   necessity                 0.31
# ordinariness              0.37   :   morality                  0.30
#
# POLITICAL_ACTION                 :   ECONOMIC_DEVELOPMENT          (nn_weighted_adjective_noun_identity)
# action                    0.50   :   commerce                  0.38
# civility                  0.34   :   completeness              0.29
# moderation                0.32   :   auspiciousness            0.28
# nastiness                 0.29   :   practicality              0.28
# fairness                  0.29   :   attractiveness            0.28
#
# OLD_PERSON                       :   ELDERLY_LADY                  (nn_weighted_adjective_noun_identity)
# AGE                       0.45   :   commonness                0.31
# otherness                 0.40   :   AGE                       0.30
# typicality                0.40   :   health                    0.29
# abstractness              0.37   :   thoughtfulness            0.29
# originality               0.36   :   kindness                  0.29
#
# EARLIER_WORK                     :   EARLY_STAGE                   (nn_weighted_adjective_noun_identity)
# TIMING                    0.34   :   TIMING                    0.61
# regularity                0.33   :   height                    0.37
# accuracy                  0.33   :   maturity                  0.30
# otherness                 0.33   :   nature                    0.29
# ordinariness              0.32   :   seriousness               0.28
#
# DIFFERENT_KIND                   :   VARIOUS_FORM                  (nn_weighted_adjective_noun_identity)
# SAMENESS                  0.57   :   OTHERNESS                 0.39
# similarity                0.54   :   COMMONALITY               0.34
# COMMONALITY               0.50   :   foreignness               0.33
# DIFFERENCE                0.50   :   SAMENESS                  0.33
# OTHERNESS                 0.48   :   DIFFERENCE                0.33
#
# LARGE_QUANTITY                   :   GREAT_MAJORITY                (nn_weighted_adjective_noun_identity)
# quantity                  0.78   :   majority                  0.70
# SIZE                      0.63   :   quality                   0.41
# width                     0.40   :   minority                  0.41
# thickness                 0.38   :   SIZE                      0.32
# volume                    0.36   :   cleanness                 0.32
#
# EFFECTIVE_WAY                    :   EFFICIENT_USE                 (nn_weighted_adjective_noun_identity)
# EFFECTIVENESS             0.75   :   EFFECTIVENESS             0.48
# efficacy                  0.48   :   PRACTICALITY              0.39
# potency                   0.44   :   ACCURACY                  0.38
# PRACTICALITY              0.43   :   quality                   0.34
# ACCURACY                  0.41   :   utility                   0.34
#
# VAST_AMOUNT                      :   HIGH_PRICE                    (nn_weighted_adjective_noun_identity)
# QUANTITY                  0.58   :   degree                    0.54
# size                      0.57   :   height                    0.48
# width                     0.42   :   price                     0.46
# thickness                 0.38   :   moderation                0.44
# substantiality            0.35   :   QUANTITY                  0.33
#
# OLD_PERSON                       :   ELDERLY_LADY                  (nn_weighted_adjective_noun_identity)
# AGE                       0.45   :   commonness                0.31
# otherness                 0.40   :   AGE                       0.30
# typicality                0.40   :   health                    0.29
# abstractness              0.37   :   thoughtfulness            0.29
# originality               0.36   :   kindness                  0.29
#
# GENERAL_PRINCIPLE                :   BASIC_RULE                    (nn_weighted_adjective_noun_identity)
# GENERALITY                0.58   :   GENERALITY                0.44
# concreteness              0.35   :   humaneness                0.43
# equality                  0.33   :   practicality              0.43
# humanness                 0.33   :   comprehensiveness         0.41
# morality                  0.33   :   reasonableness            0.41
#
# EFFECTIVE_WAY                    :   EFFICIENT_USE                 (nn_weighted_adjective_noun_identity)
# EFFECTIVENESS             0.75   :   EFFECTIVENESS             0.48
# efficacy                  0.48   :   PRACTICALITY              0.39
# potency                   0.44   :   ACCURACY                  0.38
# PRACTICALITY              0.43   :   quality                   0.34
# ACCURACY                  0.41   :   utility                   0.34
#
# FEDERAL_ASSEMBLY                 :   NATIONAL_GOVERNMENT           (nn_weighted_adjective_noun_identity)
# majority                  0.23   :   importance                0.35
# otherness                 0.22   :   significance              0.31
# nature                    0.22   :   stature                   0.30
# seniority                 0.21   :   seriousness               0.29
# sameness                  0.21   :   intrusiveness             0.29
#
# LARGE_QUANTITY                   :   GREAT_MAJORITY                (nn_weighted_adjective_noun_identity)
# quantity                  0.78   :   majority                  0.70
# SIZE                      0.63   :   quality                   0.41
# width                     0.40   :   minority                  0.41
# thickness                 0.38   :   SIZE                      0.32
# volume                    0.36   :   cleanness                 0.32
#
# CERTAIN_CIRCUMSTANCE             :   PARTICULAR_CASE               (nn_weighted_adjective_noun_identity)
# COMMONNESS                0.42   :   significance              0.49
# otherness                 0.40   :   importance                0.45
# foreignness               0.39   :   COMMONNESS                0.43
# concreteness              0.37   :   abstractness              0.41
# ordinariness              0.37   :   seriousness               0.40
#
# EARLIER_WORK                     :   EARLY_STAGE                   (nn_weighted_adjective_noun_identity)
# TIMING                    0.34   :   TIMING                    0.61
# regularity                0.33   :   height                    0.37
# accuracy                  0.33   :   maturity                  0.30
# otherness                 0.33   :   nature                    0.29
# ordinariness              0.32   :   seriousness               0.28
#
# DIFFERENT_KIND                   :   VARIOUS_FORM                  (nn_weighted_adjective_noun_identity)
# SAMENESS                  0.57   :   OTHERNESS                 0.39
# similarity                0.54   :   COMMONALITY               0.34
# COMMONALITY               0.50   :   foreignness               0.33
# DIFFERENCE                0.50   :   SAMENESS                  0.33
# OTHERNESS                 0.48   :   DIFFERENCE                0.33
#
# CENTRAL_AUTHORITY                :   LOCAL_OFFICE                  (nn_weighted_adjective_noun_identity)
# importance                0.39   :   handiness                 0.33
# significance              0.38   :   cleanness                 0.30
# power                     0.32   :   immediacy                 0.30
# potency                   0.31   :   freshness                 0.30
# responsibility            0.30   :   friendliness              0.29
#
# NEW_LIFE                         :   EARLY_AGE                     (nn_weighted_adjective_noun_identity)
# freshness                 0.43   :   age                       0.62
# permanence                0.43   :   timing                    0.58
# individuality             0.43   :   height                    0.33
# originality               0.42   :   maturity                  0.32
# dullness                  0.42   :   duration                  0.32
#
# POLITICAL_ACTION                 :   ECONOMIC_DEVELOPMENT          (nn_weighted_adjective_noun_identity)
# action                    0.50   :   commerce                  0.38
# civility                  0.34   :   completeness              0.29
# moderation                0.32   :   auspiciousness            0.28
# nastiness                 0.29   :   practicality              0.28
# fairness                  0.29   :   attractiveness            0.28
#
# FEDERAL_ASSEMBLY                 :   NATIONAL_GOVERNMENT           (nn_weighted_adjective_noun_identity)
# majority                  0.23   :   importance                0.35
# otherness                 0.22   :   significance              0.31
# nature                    0.22   :   stature                   0.30
# seniority                 0.21   :   seriousness               0.29
# sameness                  0.21   :   intrusiveness             0.29
#
# NEW_LIFE                         :   ECONOMIC_DEVELOPMENT          (nn_weighted_adjective_noun_identity)
# freshness                 0.43   :   commerce                  0.38
# permanence                0.43   :   completeness              0.29
# individuality             0.43   :   auspiciousness            0.28
# originality               0.42   :   practicality              0.28
# dullness                  0.42   :   attractiveness            0.28
#
# SMALL_HOUSE                      :   LITTLE_ROOM                   (nn_weighted_adjective_noun_identity)
# SIZE                      0.77   :   SIZE                      0.48
# HEIGHT                    0.50   :   HEIGHT                    0.44
# width                     0.46   :   niceness                  0.42
# length                    0.39   :   cleanness                 0.42
# stature                   0.39   :   hardness                  0.41
#
# NEW_BODY                         :   WHOLE_SYSTEM                  (nn_weighted_adjective_noun_identity)
# FRESHNESS                 0.34   :   integrity                 0.58
# originality               0.33   :   quality                   0.41
# cleanness                 0.33   :   fairness                  0.39
# permanence                0.32   :   honesty                   0.39
# abstractness              0.31   :   FRESHNESS                 0.38
#
# SPECIAL_CIRCUMSTANCE             :   PARTICULAR_CASE               (nn_weighted_adjective_noun_identity)
# ordinariness              0.47   :   significance              0.49
# COMMONNESS                0.44   :   importance                0.45
# otherness                 0.40   :   COMMONNESS                0.43
# nobility                  0.37   :   abstractness              0.41
# gregariousness            0.37   :   seriousness               0.40
#
# BETTER_JOB                       :   GOOD_EFFECT                   (nn_weighted_adjective_noun_identity)
# ease                      0.41   :   QUALITY                   0.50
# QUALITY                   0.40   :   effectiveness             0.43
# difficulty                0.40   :   friendliness              0.37
# naturalness               0.40   :   worthiness                0.37
# perfection                0.39   :   freshness                 0.37
#
# IMPORTANT_PART                   :   SIGNIFICANT_ROLE              (nn_weighted_adjective_noun_identity)
# IMPORTANCE                0.62   :   SIGNIFICANCE              0.56
# SIGNIFICANCE              0.54   :   IMPORTANCE                0.53
# necessity                 0.41   :   status                    0.40
# pride                     0.39   :   RESPONSIBILITY            0.36
# RESPONSIBILITY            0.35   :   likelihood                0.36
#
# NEW_LAW                          :   BASIC_RULE                    (nn_weighted_adjective_noun_identity)
# conventionality           0.35   :   generality                0.44
# abstractness              0.34   :   humaneness                0.43
# originality               0.34   :   practicality              0.43
# liveliness                0.33   :   comprehensiveness         0.41
# individuality             0.33   :   reasonableness            0.41
#
# NEW_SITUATION                    :   PRESENT_POSITION              (nn_weighted_adjective_noun_identity)
# difficulty                0.47   :   position                  0.61
# complexity                0.43   :   presence                  0.38
# abstractness              0.42   :   strength                  0.31
# concreteness              0.40   :   otherness                 0.31
# necessity                 0.39   :   stature                   0.30
#
# EARLIER_WORK                     :   EARLY_STAGE                   (nn_weighted_adjective_noun_identity)
# TIMING                    0.34   :   TIMING                    0.61
# regularity                0.33   :   height                    0.37
# accuracy                  0.33   :   maturity                  0.30
# otherness                 0.33   :   nature                    0.29
# ordinariness              0.32   :   seriousness               0.28
#
# EFFECTIVE_WAY                    :   EFFICIENT_USE                 (nn_weighted_adjective_noun_identity)
# EFFECTIVENESS             0.75   :   EFFECTIVENESS             0.48
# efficacy                  0.48   :   PRACTICALITY              0.39
# potency                   0.44   :   ACCURACY                  0.38
# PRACTICALITY              0.43   :   quality                   0.34
# ACCURACY                  0.41   :   utility                   0.34
#
# NEW_LIFE                         :   EARLY_AGE                     (nn_weighted_adjective_noun_identity)
# freshness                 0.43   :   age                       0.62
# permanence                0.43   :   timing                    0.58
# individuality             0.43   :   height                    0.33
# originality               0.42   :   maturity                  0.32
# dullness                  0.42   :   duration                  0.32
#
# LARGE_QUANTITY                   :   GREAT_MAJORITY                (nn_weighted_adjective_noun_identity)
# quantity                  0.78   :   majority                  0.70
# SIZE                      0.63   :   quality                   0.41
# width                     0.40   :   minority                  0.41
# thickness                 0.38   :   SIZE                      0.32
# volume                    0.36   :   cleanness                 0.32
# """
#
#
# test = """
# AST_AMOUNT                      :   LARGE_QUANTITY                (add)
# QUANTITY                  0.50   :   QUANTITY                  0.83
# SIZE                      0.42   :   SIZE                      0.51
# complexity                0.35   :   volume                    0.37
# SUBSTANTIALITY            0.33   :   quality                   0.36
# nature                    0.33   :   SUBSTANTIALITY            0.35
#
# VAST_AMOUNT                      :   LARGE_QUANTITY                (add)
# QUANTITY                  0.50   :   QUANTITY                  0.83
# SIZE                      0.42   :   SIZE                      0.51
# complexity                0.35   :   volume                    0.37
# SUBSTANTIALITY            0.33   :   quality                   0.36
# nature                    0.33   :   SUBSTANTIALITY            0.35
#
# IMPORTANT_PART                   :   SIGNIFICANT_ROLE              (add)
# IMPORTANCE                0.49   :   SIGNIFICANCE              0.41
# SIGNIFICANCE              0.42   :   IMPORTANCE                0.40
# good                      0.37   :   position                  0.39
# necessity                 0.37   :   RESPONSIBILITY            0.39
# RESPONSIBILITY            0.34   :   potential                 0.36
#
# LARGE_NUMBER                     :   VAST_AMOUNT                   (add)
# SIZE                      0.50   :   QUANTITY                  0.50
# majority                  0.40   :   SIZE                      0.42
# QUANTITY                  0.38   :   complexity                0.35
# volume                    0.34   :   substantiality            0.33
# potential                 0.31   :   nature                    0.33
#
# GENERAL_PRINCIPLE                :   BASIC_RULE                    (add)
# FAIRNESS                  0.40   :   MORALITY                  0.36
# MORALITY                  0.36   :   MANDATE                   0.36
# NECESSITY                 0.36   :   FAIRNESS                  0.35
# MANDATE                   0.33   :   dispensability            0.35
# majority                  0.32   :   NECESSITY                 0.31
#
# EFFECTIVE_WAY                    :   EFFICIENT_USE                 (add)
# EFFECTIVENESS             0.44   :   EFFECTIVENESS             0.34
# good                      0.36   :   capability                0.31
# ability                   0.29   :   tractability              0.29
# NECESSITY                 0.28   :   utility                   0.29
# difference                0.27   :   NECESSITY                 0.28
#
# FEDERAL_ASSEMBLY                 :   NATIONAL_GOVERNMENT           (add)
# MANDATE                   0.28   :   MANDATE                   0.36
# majority                  0.25   :   health                    0.31
# seniority                 0.24   :   independence              0.30
# MINORITY                  0.22   :   crisis                    0.28
# function                  0.22   :   MINORITY                  0.28
#
# CERTAIN_CIRCUMSTANCE             :   PARTICULAR_CASE               (add)
# timing                    0.37   :   substantiality            0.36
# nature                    0.35   :   typicality                0.36
# COMMONNESS                0.32   :   seriousness               0.36
# finality                  0.32   :   fairness                  0.34
# corruptness               0.31   :   COMMONNESS                0.34
#
# VAST_AMOUNT                      :   HIGH_PRICE                    (add)
# QUANTITY                  0.50   :   price                     0.82
# size                      0.42   :   quality                   0.34
# complexity                0.35   :   volume                    0.30
# substantiality            0.33   :   QUANTITY                  0.29
# nature                    0.33   :   temperature               0.28
#
# IMPORTANT_PART                   :   SIGNIFICANT_ROLE              (add)
# IMPORTANCE                0.49   :   SIGNIFICANCE              0.41
# SIGNIFICANCE              0.42   :   IMPORTANCE                0.40
# good                      0.37   :   position                  0.39
# necessity                 0.37   :   RESPONSIBILITY            0.39
# RESPONSIBILITY            0.34   :   potential                 0.36
#
# LARGE_NUMBER                     :   VAST_AMOUNT                   (add)
# SIZE                      0.50   :   QUANTITY                  0.50
# majority                  0.40   :   SIZE                      0.42
# QUANTITY                  0.38   :   complexity                0.35
# volume                    0.34   :   substantiality            0.33
# potential                 0.31   :   nature                    0.33
#
# FEDERAL_ASSEMBLY                 :   NATIONAL_GOVERNMENT           (add)
# MANDATE                   0.28   :   MANDATE                   0.36
# majority                  0.25   :   health                    0.31
# seniority                 0.24   :   independence              0.30
# MINORITY                  0.22   :   crisis                    0.28
# function                  0.22   :   MINORITY                  0.28
#
# IMPORTANT_PART                   :   SIGNIFICANT_ROLE              (add)
# IMPORTANCE                0.49   :   SIGNIFICANCE              0.41
# SIGNIFICANCE              0.42   :   IMPORTANCE                0.40
# good                      0.37   :   position                  0.39
# necessity                 0.37   :   RESPONSIBILITY            0.39
# RESPONSIBILITY            0.34   :   potential                 0.36
#
# CERTAIN_CIRCUMSTANCE             :   PARTICULAR_CASE               (add)
# timing                    0.37   :   substantiality            0.36
# nature                    0.35   :   typicality                0.36
# COMMONNESS                0.32   :   seriousness               0.36
# finality                  0.32   :   fairness                  0.34
# corruptness               0.31   :   COMMONNESS                0.34
#
# IMPORTANT_PART                   :   SIGNIFICANT_ROLE              (add)
# IMPORTANCE                0.49   :   SIGNIFICANCE              0.41
# SIGNIFICANCE              0.42   :   IMPORTANCE                0.40
# good                      0.37   :   position                  0.39
# necessity                 0.37   :   RESPONSIBILITY            0.39
# RESPONSIBILITY            0.34   :   potential                 0.36
#
# LARGE_NUMBER                     :   VAST_AMOUNT                   (add)
# SIZE                      0.50   :   QUANTITY                  0.50
# majority                  0.40   :   SIZE                      0.42
# QUANTITY                  0.38   :   complexity                0.35
# volume                    0.34   :   substantiality            0.33
# potential                 0.31   :   nature                    0.33
#
# GENERAL_PRINCIPLE                :   BASIC_RULE                    (add)
# FAIRNESS                  0.40   :   MORALITY                  0.36
# MORALITY                  0.36   :   MANDATE                   0.36
# NECESSITY                 0.36   :   FAIRNESS                  0.35
# MANDATE                   0.33   :   dispensability            0.35
# majority                  0.32   :   NECESSITY                 0.31
#
# DIFFERENT_KIND                   :   VARIOUS_FORM                  (add)
# good                      0.44   :   shape                     0.35
# commonality               0.36   :   regularity                0.28
# similarity                0.34   :   nature                    0.27
# sameness                  0.34   :   consistency               0.26
# ABSTRACTNESS              0.33   :   ABSTRACTNESS              0.26
#
# VAST_AMOUNT                      :   LARGE_QUANTITY                (add)
# QUANTITY                  0.50   :   QUANTITY                  0.83
# SIZE                      0.42   :   SIZE                      0.51
# complexity                0.35   :   volume                    0.37
# SUBSTANTIALITY            0.33   :   quality                   0.36
# nature                    0.33   :   SUBSTANTIALITY            0.35
#
# VAST_AMOUNT                      :   LARGE_QUANTITY                (add)
# QUANTITY                  0.50   :   QUANTITY                  0.83
# SIZE                      0.42   :   SIZE                      0.51
# complexity                0.35   :   volume                    0.37
# SUBSTANTIALITY            0.33   :   quality                   0.36
# nature                    0.33   :   SUBSTANTIALITY            0.35
#
# SOCIAL_EVENT                     :   SPECIAL_CIRCUMSTANCE          (add)
# sociality                 0.36   :   timing                    0.29
# sociability               0.34   :   significance              0.28
# awareness                 0.30   :   nature                    0.27
# equality                  0.29   :   magnitude                 0.26
# literacy                  0.27   :   auspiciousness            0.26
#
# LARGE_NUMBER                     :   VAST_AMOUNT                   (add)
# SIZE                      0.50   :   QUANTITY                  0.50
# majority                  0.40   :   SIZE                      0.42
# QUANTITY                  0.38   :   complexity                0.35
# volume                    0.34   :   substantiality            0.33
# potential                 0.31   :   nature                    0.33
#
# GENERAL_PRINCIPLE                :   BASIC_RULE                    (add)
# FAIRNESS                  0.40   :   MORALITY                  0.36
# MORALITY                  0.36   :   MANDATE                   0.36
# NECESSITY                 0.36   :   FAIRNESS                  0.35
# MANDATE                   0.33   :   dispensability            0.35
# majority                  0.32   :   NECESSITY                 0.31
#
# FEDERAL_ASSEMBLY                 :   NATIONAL_GOVERNMENT           (add)
# MANDATE                   0.28   :   MANDATE                   0.36
# majority                  0.25   :   health                    0.31
# seniority                 0.24   :   independence              0.30
# MINORITY                  0.22   :   crisis                    0.28
# function                  0.22   :   MINORITY                  0.28
#
# VAST_AMOUNT                      :   LARGE_QUANTITY                (add)
# QUANTITY                  0.50   :   QUANTITY                  0.83
# SIZE                      0.42   :   SIZE                      0.51
# complexity                0.35   :   volume                    0.37
# SUBSTANTIALITY            0.33   :   quality                   0.36
# nature                    0.33   :   SUBSTANTIALITY            0.35
#
# SPECIAL_CIRCUMSTANCE             :   PARTICULAR_CASE               (add)
# timing                    0.29   :   substantiality            0.36
# significance              0.28   :   typicality                0.36
# nature                    0.27   :   seriousness               0.36
# magnitude                 0.26   :   fairness                  0.34
# auspiciousness            0.26   :   commonness                0.34
#
# VAST_AMOUNT                      :   LARGE_QUANTITY                (add)
# QUANTITY                  0.50   :   QUANTITY                  0.83
# SIZE                      0.42   :   SIZE                      0.51
# complexity                0.35   :   volume                    0.37
# SUBSTANTIALITY            0.33   :   quality                   0.36
# nature                    0.33   :   SUBSTANTIALITY            0.35
#
# LARGE_NUMBER                     :   VAST_AMOUNT                   (add)
# SIZE                      0.50   :   QUANTITY                  0.50
# majority                  0.40   :   SIZE                      0.42
# QUANTITY                  0.38   :   complexity                0.35
# volume                    0.34   :   substantiality            0.33
# potential                 0.31   :   nature                    0.33
#
# OLD_PERSON                       :   ELDERLY_LADY                  (add)
# AGE                       0.43   :   KINDNESS                  0.28
# being                     0.23   :   AGE                       0.25
# KINDNESS                  0.22   :   health                    0.25
# connection                0.22   :   abstemiousness            0.24
# commonness                0.20   :   fertility                 0.22
#
# DIFFERENT_KIND                   :   VARIOUS_FORM                  (add)
# good                      0.44   :   shape                     0.35
# commonality               0.36   :   regularity                0.28
# similarity                0.34   :   nature                    0.27
# sameness                  0.34   :   consistency               0.26
# ABSTRACTNESS              0.33   :   ABSTRACTNESS              0.26
#
# LARGE_QUANTITY                   :   GREAT_MAJORITY                (add)
# quantity                  0.83   :   majority                  0.73
# size                      0.51   :   good                      0.59
# volume                    0.37   :   minority                  0.45
# quality                   0.36   :   pride                     0.33
# substantiality            0.35   :   continuity                0.29
#
# MAJOR_ISSUE                      :   AMERICAN_COUNTRY              (add)
# possibility               0.33   :   corruptness               0.30
# crisis                    0.30   :   freedom                   0.23
# significance              0.30   :   evil                      0.23
# importance                0.30   :   activeness                0.21
# legality                  0.29   :   repute                    0.20
#
# LARGE_NUMBER                     :   GREAT_MAJORITY                (add)
# size                      0.50   :   MAJORITY                  0.73
# MAJORITY                  0.40   :   good                      0.59
# quantity                  0.38   :   minority                  0.45
# volume                    0.34   :   pride                     0.33
# potential                 0.31   :   continuity                0.29
#
# CERTAIN_CIRCUMSTANCE             :   PARTICULAR_CASE               (add)
# timing                    0.37   :   substantiality            0.36
# nature                    0.35   :   typicality                0.36
# COMMONNESS                0.32   :   seriousness               0.36
# finality                  0.32   :   fairness                  0.34
# corruptness               0.31   :   COMMONNESS                0.34
#
# IMPORTANT_PART                   :   SIGNIFICANT_ROLE              (add)
# IMPORTANCE                0.49   :   SIGNIFICANCE              0.41
# SIGNIFICANCE              0.42   :   IMPORTANCE                0.40
# good                      0.37   :   position                  0.39
# necessity                 0.37   :   RESPONSIBILITY            0.39
# RESPONSIBILITY            0.34   :   potential                 0.36
#
# LARGE_NUMBER                     :   GREAT_MAJORITY                (add)
# size                      0.50   :   MAJORITY                  0.73
# MAJORITY                  0.40   :   good                      0.59
# quantity                  0.38   :   minority                  0.45
# volume                    0.34   :   pride                     0.33
# potential                 0.31   :   continuity                0.29
#
# CERTAIN_CIRCUMSTANCE             :   PARTICULAR_CASE               (add)
# timing                    0.37   :   substantiality            0.36
# nature                    0.35   :   typicality                0.36
# COMMONNESS                0.32   :   seriousness               0.36
# finality                  0.32   :   fairness                  0.34
# corruptness               0.31   :   COMMONNESS                0.34
#
# IMPORTANT_PART                   :   SIGNIFICANT_ROLE              (add)
# IMPORTANCE                0.49   :   SIGNIFICANCE              0.41
# SIGNIFICANCE              0.42   :   IMPORTANCE                0.40
# good                      0.37   :   position                  0.39
# necessity                 0.37   :   RESPONSIBILITY            0.39
# RESPONSIBILITY            0.34   :   potential                 0.36
#
# IMPORTANT_PART                   :   SIGNIFICANT_ROLE              (add)
# IMPORTANCE                0.49   :   SIGNIFICANCE              0.41
# SIGNIFICANCE              0.42   :   IMPORTANCE                0.40
# good                      0.37   :   position                  0.39
# necessity                 0.37   :   RESPONSIBILITY            0.39
# RESPONSIBILITY            0.34   :   potential                 0.36
#
# LARGE_NUMBER                     :   VAST_AMOUNT                   (add)
# SIZE                      0.50   :   QUANTITY                  0.50
# majority                  0.40   :   SIZE                      0.42
# QUANTITY                  0.38   :   complexity                0.35
# volume                    0.34   :   substantiality            0.33
# potential                 0.31   :   nature                    0.33
#
# LARGE_NUMBER                     :   VAST_AMOUNT                   (add)
# SIZE                      0.50   :   QUANTITY                  0.50
# majority                  0.40   :   SIZE                      0.42
# QUANTITY                  0.38   :   complexity                0.35
# volume                    0.34   :   substantiality            0.33
# potential                 0.31   :   nature                    0.33
#
# VAST_AMOUNT                      :   LARGE_QUANTITY                (add)
# QUANTITY                  0.50   :   QUANTITY                  0.83
# SIZE                      0.42   :   SIZE                      0.51
# complexity                0.35   :   volume                    0.37
# SUBSTANTIALITY            0.33   :   quality                   0.36
# nature                    0.33   :   SUBSTANTIALITY            0.35
#
# VAST_AMOUNT                      :   LARGE_QUANTITY                (add)
# QUANTITY                  0.50   :   QUANTITY                  0.83
# SIZE                      0.42   :   SIZE                      0.51
# complexity                0.35   :   volume                    0.37
# SUBSTANTIALITY            0.33   :   quality                   0.36
# nature                    0.33   :   SUBSTANTIALITY            0.35
#
# NEW_INFORMATION                  :   FURTHER_EVIDENCE              (add)
# reassurance               0.28   :   likelihood                0.37
# intelligence              0.27   :   possibility               0.36
# convenience               0.26   :   credibility               0.33
# excitement                0.26   :   substantiality            0.31
# clarity                   0.25   :   truth                     0.30
#
# VAST_AMOUNT                      :   LARGE_QUANTITY                (add)
# QUANTITY                  0.50   :   QUANTITY                  0.83
# SIZE                      0.42   :   SIZE                      0.51
# complexity                0.35   :   volume                    0.37
# SUBSTANTIALITY            0.33   :   quality                   0.36
# nature                    0.33   :   SUBSTANTIALITY            0.35
#
# LARGE_NUMBER                     :   VAST_AMOUNT                   (add)
# SIZE                      0.50   :   QUANTITY                  0.50
# majority                  0.40   :   SIZE                      0.42
# QUANTITY                  0.38   :   complexity                0.35
# volume                    0.34   :   substantiality            0.33
# potential                 0.31   :   nature                    0.33
#
# LARGE_QUANTITY                   :   GREAT_MAJORITY                (add)
# quantity                  0.83   :   majority                  0.73
# size                      0.51   :   good                      0.59
# volume                    0.37   :   minority                  0.45
# quality                   0.36   :   pride                     0.33
# substantiality            0.35   :   continuity                0.29
#
# LARGE_NUMBER                     :   GREAT_MAJORITY                (add)
# size                      0.50   :   MAJORITY                  0.73
# MAJORITY                  0.40   :   good                      0.59
# quantity                  0.38   :   minority                  0.45
# volume                    0.34   :   pride                     0.33
# potential                 0.31   :   continuity                0.29
#
# LARGE_NUMBER                     :   VAST_AMOUNT                   (add)
# SIZE                      0.50   :   QUANTITY                  0.50
# majority                  0.40   :   SIZE                      0.42
# QUANTITY                  0.38   :   complexity                0.35
# volume                    0.34   :   substantiality            0.33
# potential                 0.31   :   nature                    0.33
#
# OLD_PERSON                       :   ELDERLY_LADY                  (add)
# AGE                       0.43   :   KINDNESS                  0.28
# being                     0.23   :   AGE                       0.25
# KINDNESS                  0.22   :   health                    0.25
# connection                0.22   :   abstemiousness            0.24
# commonness                0.20   :   fertility                 0.22
#
# EARLIER_WORK                     :   EARLY_STAGE                   (add)
# being                     0.24   :   timing                    0.27
# difficulty                0.21   :   age                       0.26
# good                      0.21   :   maturity                  0.24
# seniority                 0.20   :   height                    0.22
# thoughtfulness            0.20   :   potential                 0.21
#
# GENERAL_PRINCIPLE                :   BASIC_RULE                    (add)
# FAIRNESS                  0.40   :   MORALITY                  0.36
# MORALITY                  0.36   :   MANDATE                   0.36
# NECESSITY                 0.36   :   FAIRNESS                  0.35
# MANDATE                   0.33   :   dispensability            0.35
# majority                  0.32   :   NECESSITY                 0.31
#
# LARGE_QUANTITY                   :   GREAT_MAJORITY                (add)
# quantity                  0.83   :   majority                  0.73
# size                      0.51   :   good                      0.59
# volume                    0.37   :   minority                  0.45
# quality                   0.36   :   pride                     0.33
# substantiality            0.35   :   continuity                0.29
#
# NEW_LIFE                         :   EARLY_AGE                     (add)
# happiness                 0.34   :   age                       0.84
# humanness                 0.34   :   maturity                  0.36
# domesticity               0.32   :   height                    0.35
# freedom                   0.31   :   dormancy                  0.26
# reality                   0.30   :   susceptibility            0.25
#
# GENERAL_PRINCIPLE                :   BASIC_RULE                    (add)
# FAIRNESS                  0.40   :   MORALITY                  0.36
# MORALITY                  0.36   :   MANDATE                   0.36
# NECESSITY                 0.36   :   FAIRNESS                  0.35
# MANDATE                   0.33   :   dispensability            0.35
# majority                  0.32   :   NECESSITY                 0.31
#
# DIFFERENT_KIND                   :   VARIOUS_FORM                  (add)
# good                      0.44   :   shape                     0.35
# commonality               0.36   :   regularity                0.28
# similarity                0.34   :   nature                    0.27
# sameness                  0.34   :   consistency               0.26
# ABSTRACTNESS              0.33   :   ABSTRACTNESS              0.26
#
# CERTAIN_CIRCUMSTANCE             :   PARTICULAR_CASE               (add)
# timing                    0.37   :   substantiality            0.36
# nature                    0.35   :   typicality                0.36
# COMMONNESS                0.32   :   seriousness               0.36
# finality                  0.32   :   fairness                  0.34
# corruptness               0.31   :   COMMONNESS                0.34
#
# IMPORTANT_PART                   :   SIGNIFICANT_ROLE              (add)
# IMPORTANCE                0.49   :   SIGNIFICANCE              0.41
# SIGNIFICANCE              0.42   :   IMPORTANCE                0.40
# good                      0.37   :   position                  0.39
# necessity                 0.37   :   RESPONSIBILITY            0.39
# RESPONSIBILITY            0.34   :   potential                 0.36
#
# VAST_AMOUNT                      :   LARGE_QUANTITY                (add)
# QUANTITY                  0.50   :   QUANTITY                  0.83
# SIZE                      0.42   :   SIZE                      0.51
# complexity                0.35   :   volume                    0.37
# SUBSTANTIALITY            0.33   :   quality                   0.36
# nature                    0.33   :   SUBSTANTIALITY            0.35
#
# LARGE_NUMBER                     :   VAST_AMOUNT                   (add)
# SIZE                      0.50   :   QUANTITY                  0.50
# majority                  0.40   :   SIZE                      0.42
# QUANTITY                  0.38   :   complexity                0.35
# volume                    0.34   :   substantiality            0.33
# potential                 0.31   :   nature                    0.33
#
# GENERAL_PRINCIPLE                :   BASIC_RULE                    (add)
# FAIRNESS                  0.40   :   MORALITY                  0.36
# MORALITY                  0.36   :   MANDATE                   0.36
# NECESSITY                 0.36   :   FAIRNESS                  0.35
# MANDATE                   0.33   :   dispensability            0.35
# majority                  0.32   :   NECESSITY                 0.31
#
# DIFFERENT_KIND                   :   VARIOUS_FORM                  (add)
# good                      0.44   :   shape                     0.35
# commonality               0.36   :   regularity                0.28
# similarity                0.34   :   nature                    0.27
# sameness                  0.34   :   consistency               0.26
# ABSTRACTNESS              0.33   :   ABSTRACTNESS              0.26
#
# FEDERAL_ASSEMBLY                 :   NATIONAL_GOVERNMENT           (add)
# MANDATE                   0.28   :   MANDATE                   0.36
# majority                  0.25   :   health                    0.31
# seniority                 0.24   :   independence              0.30
# MINORITY                  0.22   :   crisis                    0.28
# function                  0.22   :   MINORITY                  0.28
#
# LARGE_NUMBER                     :   VAST_AMOUNT                   (add)
# SIZE                      0.50   :   QUANTITY                  0.50
# majority                  0.40   :   SIZE                      0.42
# QUANTITY                  0.38   :   complexity                0.35
# volume                    0.34   :   substantiality            0.33
# potential                 0.31   :   nature                    0.33
#
# OLD_PERSON                       :   ELDERLY_LADY                  (add)
# AGE                       0.43   :   KINDNESS                  0.28
# being                     0.23   :   AGE                       0.25
# KINDNESS                  0.22   :   health                    0.25
# connection                0.22   :   abstemiousness            0.24
# commonness                0.20   :   fertility                 0.22
#
# NEW_INFORMATION                  :   FURTHER_EVIDENCE              (add)
# reassurance               0.28   :   likelihood                0.37
# intelligence              0.27   :   possibility               0.36
# convenience               0.26   :   credibility               0.33
# excitement                0.26   :   substantiality            0.31
# clarity                   0.25   :   truth                     0.30
#
# SPECIAL_CIRCUMSTANCE             :   PARTICULAR_CASE               (add)
# timing                    0.29   :   substantiality            0.36
# significance              0.28   :   typicality                0.36
# nature                    0.27   :   seriousness               0.36
# magnitude                 0.26   :   fairness                  0.34
# auspiciousness            0.26   :   commonness                0.34
#
# VAST_AMOUNT                      :   LARGE_QUANTITY                (add)
# QUANTITY                  0.50   :   QUANTITY                  0.83
# SIZE                      0.42   :   SIZE                      0.51
# complexity                0.35   :   volume                    0.37
# SUBSTANTIALITY            0.33   :   quality                   0.36
# nature                    0.33   :   SUBSTANTIALITY            0.35
#
# VAST_AMOUNT                      :   LARGE_QUANTITY                (add)
# QUANTITY                  0.50   :   QUANTITY                  0.83
# SIZE                      0.42   :   SIZE                      0.51
# complexity                0.35   :   volume                    0.37
# SUBSTANTIALITY            0.33   :   quality                   0.36
# nature                    0.33   :   SUBSTANTIALITY            0.35
#
# NEW_INFORMATION                  :   FURTHER_EVIDENCE              (add)
# reassurance               0.28   :   likelihood                0.37
# intelligence              0.27   :   possibility               0.36
# convenience               0.26   :   credibility               0.33
# excitement                0.26   :   substantiality            0.31
# clarity                   0.25   :   truth                     0.30
#
# VAST_AMOUNT                      :   LARGE_QUANTITY                (add)
# QUANTITY                  0.50   :   QUANTITY                  0.83
# SIZE                      0.42   :   SIZE                      0.51
# complexity                0.35   :   volume                    0.37
# SUBSTANTIALITY            0.33   :   quality                   0.36
# nature                    0.33   :   SUBSTANTIALITY            0.35
#
# LARGE_NUMBER                     :   VAST_AMOUNT                   (add)
# SIZE                      0.50   :   QUANTITY                  0.50
# majority                  0.40   :   SIZE                      0.42
# QUANTITY                  0.38   :   complexity                0.35
# volume                    0.34   :   substantiality            0.33
# potential                 0.31   :   nature                    0.33
#
# OLD_PERSON                       :   ELDERLY_LADY                  (add)
# AGE                       0.43   :   KINDNESS                  0.28
# being                     0.23   :   AGE                       0.25
# KINDNESS                  0.22   :   health                    0.25
# connection                0.22   :   abstemiousness            0.24
# commonness                0.20   :   fertility                 0.22
#
# DIFFERENT_KIND                   :   VARIOUS_FORM                  (add)
# good                      0.44   :   shape                     0.35
# commonality               0.36   :   regularity                0.28
# similarity                0.34   :   nature                    0.27
# sameness                  0.34   :   consistency               0.26
# ABSTRACTNESS              0.33   :   ABSTRACTNESS              0.26
#
# GENERAL_PRINCIPLE                :   BASIC_RULE                    (add)
# FAIRNESS                  0.40   :   MORALITY                  0.36
# MORALITY                  0.36   :   MANDATE                   0.36
# NECESSITY                 0.36   :   FAIRNESS                  0.35
# MANDATE                   0.33   :   dispensability            0.35
# majority                  0.32   :   NECESSITY                 0.31
#
# LARGE_NUMBER                     :   VAST_AMOUNT                   (add)
# SIZE                      0.50   :   QUANTITY                  0.50
# majority                  0.40   :   SIZE                      0.42
# QUANTITY                  0.38   :   complexity                0.35
# volume                    0.34   :   substantiality            0.33
# potential                 0.31   :   nature                    0.33
#
# GENERAL_PRINCIPLE                :   BASIC_RULE                    (add)
# FAIRNESS                  0.40   :   MORALITY                  0.36
# MORALITY                  0.36   :   MANDATE                   0.36
# NECESSITY                 0.36   :   FAIRNESS                  0.35
# MANDATE                   0.33   :   dispensability            0.35
# majority                  0.32   :   NECESSITY                 0.31
#
# LARGE_QUANTITY                   :   GREAT_MAJORITY                (add)
# quantity                  0.83   :   majority                  0.73
# size                      0.51   :   good                      0.59
# volume                    0.37   :   minority                  0.45
# quality                   0.36   :   pride                     0.33
# substantiality            0.35   :   continuity                0.29
#
# IMPORTANT_PART                   :   SIGNIFICANT_ROLE              (add)
# IMPORTANCE                0.49   :   SIGNIFICANCE              0.41
# SIGNIFICANCE              0.42   :   IMPORTANCE                0.40
# good                      0.37   :   position                  0.39
# necessity                 0.37   :   RESPONSIBILITY            0.39
# RESPONSIBILITY            0.34   :   potential                 0.36
#
# NEW_INFORMATION                  :   FURTHER_EVIDENCE              (add)
# reassurance               0.28   :   likelihood                0.37
# intelligence              0.27   :   possibility               0.36
# convenience               0.26   :   credibility               0.33
# excitement                0.26   :   substantiality            0.31
# clarity                   0.25   :   truth                     0.30
#
# VAST_AMOUNT                      :   LARGE_QUANTITY                (add)
# QUANTITY                  0.50   :   QUANTITY                  0.83
# SIZE                      0.42   :   SIZE                      0.51
# complexity                0.35   :   volume                    0.37
# SUBSTANTIALITY            0.33   :   quality                   0.36
# nature                    0.33   :   SUBSTANTIALITY            0.35
#
# LARGE_NUMBER                     :   GREAT_MAJORITY                (add)
# size                      0.50   :   MAJORITY                  0.73
# MAJORITY                  0.40   :   good                      0.59
# quantity                  0.38   :   minority                  0.45
# volume                    0.34   :   pride                     0.33
# potential                 0.31   :   continuity                0.29
#
# SPECIAL_CIRCUMSTANCE             :   PARTICULAR_CASE               (add)
# timing                    0.29   :   substantiality            0.36
# significance              0.28   :   typicality                0.36
# nature                    0.27   :   seriousness               0.36
# magnitude                 0.26   :   fairness                  0.34
# auspiciousness            0.26   :   commonness                0.34
#
# INDUSTRIAL_AREA                  :   WHOLE_COUNTRY                 (add)
# commerce                  0.27   :   good                      0.31
# utility                   0.26   :   truth                     0.28
# cleanness                 0.24   :   mind                      0.28
# potential                 0.21   :   corruptness               0.28
# action                    0.19   :   pride                     0.27
#
# SOCIAL_ACTIVITY                  :   ECONOMIC_CONDITION            (add)
# sociability               0.46   :   health                    0.42
# sociality                 0.44   :   crisis                    0.35
# activeness                0.36   :   shape                     0.32
# permissiveness            0.35   :   cyclicity                 0.29
# acquisitiveness           0.33   :   commerce                  0.27
#
# GOOD_PLACE                       :   HIGH_POINT                    (add)
# GOOD                      0.77   :   height                    0.31
# shape                     0.38   :   GOOD                      0.30
# position                  0.31   :   maturity                  0.25
# consistency               0.31   :   degree                    0.25
# pride                     0.30   :   speed                     0.24
#
# NEW_INFORMATION                  :   FURTHER_EVIDENCE              (add)
# reassurance               0.28   :   likelihood                0.37
# intelligence              0.27   :   possibility               0.36
# convenience               0.26   :   credibility               0.33
# excitement                0.26   :   substantiality            0.31
# clarity                   0.25   :   truth                     0.30
#
# DIFFERENT_PART                   :   VARIOUS_FORM                  (add)
# commonality               0.36   :   SHAPE                     0.35
# difference                0.36   :   regularity                0.28
# SHAPE                     0.35   :   nature                    0.27
# good                      0.34   :   consistency               0.26
# similarity                0.32   :   abstractness              0.26
#
# ECONOMIC_PROBLEM                 :   PRACTICAL_DIFFICULTY          (add)
# crisis                    0.55   :   DIFFICULTY                0.80
# DIFFICULTY                0.37   :   NECESSITY                 0.46
# NECESSITY                 0.34   :   practicality              0.45
# IMPORTANCE                0.30   :   complexity                0.44
# dispensability            0.30   :   IMPORTANCE                0.42
#
# EFFECTIVE_WAY                    :   IMPORTANT_PART                (add)
# effectiveness             0.44   :   importance                0.49
# GOOD                      0.36   :   significance              0.42
# ability                   0.29   :   GOOD                      0.37
# NECESSITY                 0.28   :   NECESSITY                 0.37
# difference                0.27   :   responsibility            0.34
#
# NEW_SITUATION                    :   DIFFERENT_KIND                (add)
# crisis                    0.42   :   good                      0.44
# position                  0.39   :   commonality               0.36
# possibility               0.34   :   similarity                0.34
# reality                   0.34   :   sameness                  0.34
# shape                     0.30   :   abstractness              0.33
#
# LARGE_NUMBER                     :   GREAT_MAJORITY                (add)
# size                      0.50   :   MAJORITY                  0.73
# MAJORITY                  0.40   :   good                      0.59
# quantity                  0.38   :   minority                  0.45
# volume                    0.34   :   pride                     0.33
# potential                 0.31   :   continuity                0.29
#
# CERTAIN_CIRCUMSTANCE             :   PARTICULAR_CASE               (add)
# timing                    0.37   :   substantiality            0.36
# nature                    0.35   :   typicality                0.36
# COMMONNESS                0.32   :   seriousness               0.36
# finality                  0.32   :   fairness                  0.34
# corruptness               0.31   :   COMMONNESS                0.34
#
# VAST_AMOUNT                      :   HIGH_PRICE                    (add)
# QUANTITY                  0.50   :   price                     0.82
# size                      0.42   :   quality                   0.34
# complexity                0.35   :   volume                    0.30
# substantiality            0.33   :   QUANTITY                  0.29
# nature                    0.33   :   temperature               0.28
#
# NEW_LAW                          :   BASIC_RULE                    (add)
# MANDATE                   0.36   :   MORALITY                  0.36
# legality                  0.30   :   MANDATE                   0.36
# lawfulness                0.29   :   fairness                  0.35
# MORALITY                  0.26   :   dispensability            0.35
# measure                   0.26   :   necessity                 0.31
#
# CENTRAL_AUTHORITY                :   LOCAL_OFFICE                  (add)
# MANDATE                   0.41   :   health                    0.23
# power                     0.38   :   commerce                  0.22
# RESPONSIBILITY            0.35   :   MANDATE                   0.22
# legality                  0.31   :   RESPONSIBILITY            0.19
# FUNCTION                  0.29   :   FUNCTION                  0.19
#
# LARGE_NUMBER                     :   GREAT_MAJORITY                (add)
# size                      0.50   :   MAJORITY                  0.73
# MAJORITY                  0.40   :   good                      0.59
# quantity                  0.38   :   minority                  0.45
# volume                    0.34   :   pride                     0.33
# potential                 0.31   :   continuity                0.29
#
# OLD_PERSON                       :   ELDERLY_LADY                  (add)
# AGE                       0.43   :   KINDNESS                  0.28
# being                     0.23   :   AGE                       0.25
# KINDNESS                  0.22   :   health                    0.25
# connection                0.22   :   abstemiousness            0.24
# commonness                0.20   :   fertility                 0.22
#
# GENERAL_PRINCIPLE                :   BASIC_RULE                    (add)
# FAIRNESS                  0.40   :   MORALITY                  0.36
# MORALITY                  0.36   :   MANDATE                   0.36
# NECESSITY                 0.36   :   FAIRNESS                  0.35
# MANDATE                   0.33   :   dispensability            0.35
# majority                  0.32   :   NECESSITY                 0.31
#
# LARGE_NUMBER                     :   GREAT_MAJORITY                (add)
# size                      0.50   :   MAJORITY                  0.73
# MAJORITY                  0.40   :   good                      0.59
# quantity                  0.38   :   minority                  0.45
# volume                    0.34   :   pride                     0.33
# potential                 0.31   :   continuity                0.29
#
# CERTAIN_CIRCUMSTANCE             :   PARTICULAR_CASE               (add)
# timing                    0.37   :   substantiality            0.36
# nature                    0.35   :   typicality                0.36
# COMMONNESS                0.32   :   seriousness               0.36
# finality                  0.32   :   fairness                  0.34
# corruptness               0.31   :   COMMONNESS                0.34
#
# PREVIOUS_DAY                     :   EARLY_AGE                     (add)
# duration                  0.34   :   age                       0.84
# volume                    0.25   :   maturity                  0.36
# timing                    0.25   :   height                    0.35
# mind                      0.23   :   dormancy                  0.26
# length                    0.22   :   susceptibility            0.25
#
# LARGE_NUMBER                     :   VAST_AMOUNT                   (add)
# SIZE                      0.50   :   QUANTITY                  0.50
# majority                  0.40   :   SIZE                      0.42
# QUANTITY                  0.38   :   complexity                0.35
# volume                    0.34   :   substantiality            0.33
# potential                 0.31   :   nature                    0.33
#
# EFFECTIVE_WAY                    :   EFFICIENT_USE                 (add)
# EFFECTIVENESS             0.44   :   EFFECTIVENESS             0.34
# good                      0.36   :   capability                0.31
# ability                   0.29   :   tractability              0.29
# NECESSITY                 0.28   :   utility                   0.29
# difference                0.27   :   NECESSITY                 0.28
#
# LARGE_NUMBER                     :   GREAT_MAJORITY                (add)
# size                      0.50   :   MAJORITY                  0.73
# MAJORITY                  0.40   :   good                      0.59
# quantity                  0.38   :   minority                  0.45
# volume                    0.34   :   pride                     0.33
# potential                 0.31   :   continuity                0.29
#
# RIGHT_HAND                       :   LEFT_ARM                      (add)
# DIRECTION                 0.32   :   PITCH                     0.27
# good                      0.29   :   POSITION                  0.26
# PITCH                     0.29   :   strength                  0.25
# mind                      0.26   :   length                    0.23
# POSITION                  0.25   :   DIRECTION                 0.23
#
# OLD_PERSON                       :   ELDERLY_LADY                  (add)
# AGE                       0.43   :   KINDNESS                  0.28
# being                     0.23   :   AGE                       0.25
# KINDNESS                  0.22   :   health                    0.25
# connection                0.22   :   abstemiousness            0.24
# commonness                0.20   :   fertility                 0.22
#
# FEDERAL_ASSEMBLY                 :   NATIONAL_GOVERNMENT           (add)
# MANDATE                   0.28   :   MANDATE                   0.36
# majority                  0.25   :   health                    0.31
# seniority                 0.24   :   independence              0.30
# MINORITY                  0.22   :   crisis                    0.28
# function                  0.22   :   MINORITY                  0.28
#
# BLACK_HAIR                       :   DARK_EYE                      (add)
# COMPLEXION                0.39   :   light                     0.43
# auspiciousness            0.30   :   mind                      0.37
# texture                   0.29   :   luminosity                0.33
# virginity                 0.29   :   COMPLEXION                0.30
# ancestry                  0.27   :   sharpness                 0.30
#
# SMALL_HOUSE                      :   LITTLE_ROOM                   (add)
# size                      0.35   :   comfort                   0.33
# majority                  0.25   :   good                      0.29
# quantity                  0.23   :   mind                      0.26
# minority                  0.23   :   actuality                 0.25
# ordinariness              0.19   :   light                     0.24
#
# AMERICAN_COUNTRY                 :   EUROPEAN_STATE                (add)
# CORRUPTNESS               0.30   :   typicality                0.22
# freedom                   0.23   :   function                  0.19
# evil                      0.23   :   essentiality              0.19
# activeness                0.21   :   dispensability            0.19
# repute                    0.20   :   CORRUPTNESS               0.19
#
# NEW_INFORMATION                  :   FURTHER_EVIDENCE              (add)
# reassurance               0.28   :   likelihood                0.37
# intelligence              0.27   :   possibility               0.36
# convenience               0.26   :   credibility               0.33
# excitement                0.26   :   substantiality            0.31
# clarity                   0.25   :   truth                     0.30
#
# SPECIAL_CIRCUMSTANCE             :   PARTICULAR_CASE               (add)
# timing                    0.29   :   substantiality            0.36
# significance              0.28   :   typicality                0.36
# nature                    0.27   :   seriousness               0.36
# magnitude                 0.26   :   fairness                  0.34
# auspiciousness            0.26   :   commonness                0.34
#
# DIFFERENT_PART                   :   VARIOUS_FORM                  (add)
# commonality               0.36   :   SHAPE                     0.35
# difference                0.36   :   regularity                0.28
# SHAPE                     0.35   :   nature                    0.27
# good                      0.34   :   consistency               0.26
# similarity                0.32   :   abstractness              0.26
#
# SPECIAL_CIRCUMSTANCE             :   PARTICULAR_CASE               (add)
# timing                    0.29   :   substantiality            0.36
# significance              0.28   :   typicality                0.36
# nature                    0.27   :   seriousness               0.36
# magnitude                 0.26   :   fairness                  0.34
# auspiciousness            0.26   :   commonness                0.34
#
# NEW_SITUATION                    :   DIFFERENT_KIND                (add)
# crisis                    0.42   :   good                      0.44
# position                  0.39   :   commonality               0.36
# possibility               0.34   :   similarity                0.34
# reality                   0.34   :   sameness                  0.34
# shape                     0.30   :   abstractness              0.33
#
# OLD_PERSON                       :   ELDERLY_LADY                  (add)
# AGE                       0.43   :   KINDNESS                  0.28
# being                     0.23   :   AGE                       0.25
# KINDNESS                  0.22   :   health                    0.25
# connection                0.22   :   abstemiousness            0.24
# commonness                0.20   :   fertility                 0.22
#
# EARLIER_WORK                     :   EARLY_STAGE                   (add)
# being                     0.24   :   timing                    0.27
# difficulty                0.21   :   age                       0.26
# good                      0.21   :   maturity                  0.24
# seniority                 0.20   :   height                    0.22
# thoughtfulness            0.20   :   potential                 0.21
#
# DIFFERENT_KIND                   :   VARIOUS_FORM                  (add)
# good                      0.44   :   shape                     0.35
# commonality               0.36   :   regularity                0.28
# similarity                0.34   :   nature                    0.27
# sameness                  0.34   :   consistency               0.26
# ABSTRACTNESS              0.33   :   ABSTRACTNESS              0.26
#
# EFFECTIVE_WAY                    :   EFFICIENT_USE                 (add)
# EFFECTIVENESS             0.44   :   EFFECTIVENESS             0.34
# good                      0.36   :   capability                0.31
# ability                   0.29   :   tractability              0.29
# NECESSITY                 0.28   :   utility                   0.29
# difference                0.27   :   NECESSITY                 0.28
#
# MAJOR_ISSUE                      :   AMERICAN_COUNTRY              (add)
# possibility               0.33   :   corruptness               0.30
# crisis                    0.30   :   freedom                   0.23
# significance              0.30   :   evil                      0.23
# importance                0.30   :   activeness                0.21
# legality                  0.29   :   repute                    0.20
#
# LARGE_QUANTITY                   :   GREAT_MAJORITY                (add)
# quantity                  0.83   :   majority                  0.73
# size                      0.51   :   good                      0.59
# volume                    0.37   :   minority                  0.45
# quality                   0.36   :   pride                     0.33
# substantiality            0.35   :   continuity                0.29
#
# BLACK_HAIR                       :   DARK_EYE                      (add)
# COMPLEXION                0.39   :   light                     0.43
# auspiciousness            0.30   :   mind                      0.37
# texture                   0.29   :   luminosity                0.33
# virginity                 0.29   :   COMPLEXION                0.30
# ancestry                  0.27   :   sharpness                 0.30
#
# VAST_AMOUNT                      :   LARGE_QUANTITY                (add)
# QUANTITY                  0.50   :   QUANTITY                  0.83
# SIZE                      0.42   :   SIZE                      0.51
# complexity                0.35   :   volume                    0.37
# SUBSTANTIALITY            0.33   :   quality                   0.36
# nature                    0.33   :   SUBSTANTIALITY            0.35
#
# EARLIER_WORK                     :   EARLY_STAGE                   (add)
# being                     0.24   :   timing                    0.27
# difficulty                0.21   :   age                       0.26
# good                      0.21   :   maturity                  0.24
# seniority                 0.20   :   height                    0.22
# thoughtfulness            0.20   :   potential                 0.21
#
# GENERAL_PRINCIPLE                :   BASIC_RULE                    (add)
# FAIRNESS                  0.40   :   MORALITY                  0.36
# MORALITY                  0.36   :   MANDATE                   0.36
# NECESSITY                 0.36   :   FAIRNESS                  0.35
# MANDATE                   0.33   :   dispensability            0.35
# majority                  0.32   :   NECESSITY                 0.31
#
# EFFECTIVE_WAY                    :   EFFICIENT_USE                 (add)
# EFFECTIVENESS             0.44   :   EFFECTIVENESS             0.34
# good                      0.36   :   capability                0.31
# ability                   0.29   :   tractability              0.29
# NECESSITY                 0.28   :   utility                   0.29
# difference                0.27   :   NECESSITY                 0.28
#
# OLDER_MAN                        :   ELDERLY_WOMAN                 (add)
# AGE                       0.53   :   AGE                       0.31
# ancestry                  0.25   :   kindness                  0.28
# SEX                       0.25   :   SEX                       0.27
# commonness                0.23   :   fertility                 0.27
# honesty                   0.22   :   health                    0.24
#
# FEDERAL_ASSEMBLY                 :   NATIONAL_GOVERNMENT           (add)
# MANDATE                   0.28   :   MANDATE                   0.36
# majority                  0.25   :   health                    0.31
# seniority                 0.24   :   independence              0.30
# MINORITY                  0.22   :   crisis                    0.28
# function                  0.22   :   MINORITY                  0.28
#
# VAST_AMOUNT                      :   HIGH_PRICE                    (add)
# QUANTITY                  0.50   :   price                     0.82
# size                      0.42   :   quality                   0.34
# complexity                0.35   :   volume                    0.30
# substantiality            0.33   :   QUANTITY                  0.29
# nature                    0.33   :   temperature               0.28
#
# SOCIAL_ACTIVITY                  :   POLITICAL_ACTION              (add)
# sociability               0.46   :   action                    0.70
# sociality                 0.44   :   cowardice                 0.32
# activeness                0.36   :   crisis                    0.32
# permissiveness            0.35   :   morality                  0.31
# acquisitiveness           0.33   :   civility                  0.31
#
# RIGHT_HAND                       :   LEFT_ARM                      (add)
# DIRECTION                 0.32   :   PITCH                     0.27
# good                      0.29   :   POSITION                  0.26
# PITCH                     0.29   :   strength                  0.25
# mind                      0.26   :   length                    0.23
# POSITION                  0.25   :   DIRECTION                 0.23
#
# VAST_AMOUNT                      :   HIGH_PRICE                    (add)
# QUANTITY                  0.50   :   price                     0.82
# size                      0.42   :   quality                   0.34
# complexity                0.35   :   volume                    0.30
# substantiality            0.33   :   QUANTITY                  0.29
# nature                    0.33   :   temperature               0.28
#
# OLD_PERSON                       :   ELDERLY_LADY                  (add)
# AGE                       0.43   :   KINDNESS                  0.28
# being                     0.23   :   AGE                       0.25
# KINDNESS                  0.22   :   health                    0.25
# connection                0.22   :   abstemiousness            0.24
# commonness                0.20   :   fertility                 0.22
#
# GENERAL_PRINCIPLE                :   BASIC_RULE                    (add)
# FAIRNESS                  0.40   :   MORALITY                  0.36
# MORALITY                  0.36   :   MANDATE                   0.36
# NECESSITY                 0.36   :   FAIRNESS                  0.35
# MANDATE                   0.33   :   dispensability            0.35
# majority                  0.32   :   NECESSITY                 0.31
#
# DIFFERENT_KIND                   :   VARIOUS_FORM                  (add)
# good                      0.44   :   shape                     0.35
# commonality               0.36   :   regularity                0.28
# similarity                0.34   :   nature                    0.27
# sameness                  0.34   :   consistency               0.26
# ABSTRACTNESS              0.33   :   ABSTRACTNESS              0.26
#
# EFFECTIVE_WAY                    :   EFFICIENT_USE                 (add)
# EFFECTIVENESS             0.44   :   EFFECTIVENESS             0.34
# good                      0.36   :   capability                0.31
# ability                   0.29   :   tractability              0.29
# NECESSITY                 0.28   :   utility                   0.29
# difference                0.27   :   NECESSITY                 0.28
#
# LARGE_QUANTITY                   :   GREAT_MAJORITY                (add)
# quantity                  0.83   :   majority                  0.73
# size                      0.51   :   good                      0.59
# volume                    0.37   :   minority                  0.45
# quality                   0.36   :   pride                     0.33
# substantiality            0.35   :   continuity                0.29
#
# GOOD_PLACE                       :   HIGH_POINT                    (add)
# GOOD                      0.77   :   height                    0.31
# shape                     0.38   :   GOOD                      0.30
# position                  0.31   :   maturity                  0.25
# consistency               0.31   :   degree                    0.25
# pride                     0.30   :   speed                     0.24
#
# NEW_INFORMATION                  :   FURTHER_EVIDENCE              (add)
# reassurance               0.28   :   likelihood                0.37
# intelligence              0.27   :   possibility               0.36
# convenience               0.26   :   credibility               0.33
# excitement                0.26   :   substantiality            0.31
# clarity                   0.25   :   truth                     0.30
#
# SOCIAL_ACTIVITY                  :   ECONOMIC_CONDITION            (add)
# sociability               0.46   :   health                    0.42
# sociality                 0.44   :   crisis                    0.35
# activeness                0.36   :   shape                     0.32
# permissiveness            0.35   :   cyclicity                 0.29
# acquisitiveness           0.33   :   commerce                  0.27
#
# SPECIAL_CIRCUMSTANCE             :   PARTICULAR_CASE               (add)
# timing                    0.29   :   substantiality            0.36
# significance              0.28   :   typicality                0.36
# nature                    0.27   :   seriousness               0.36
# magnitude                 0.26   :   fairness                  0.34
# auspiciousness            0.26   :   commonness                0.34
#
# LARGE_NUMBER                     :   GREAT_MAJORITY                (add)
# size                      0.50   :   MAJORITY                  0.73
# MAJORITY                  0.40   :   good                      0.59
# quantity                  0.38   :   minority                  0.45
# volume                    0.34   :   pride                     0.33
# potential                 0.31   :   continuity                0.29
#
# IMPORTANT_PART                   :   SIGNIFICANT_ROLE              (add)
# IMPORTANCE                0.49   :   SIGNIFICANCE              0.41
# SIGNIFICANCE              0.42   :   IMPORTANCE                0.40
# good                      0.37   :   position                  0.39
# necessity                 0.37   :   RESPONSIBILITY            0.39
# RESPONSIBILITY            0.34   :   potential                 0.36
#
# OLD_PERSON                       :   ELDERLY_LADY                  (add)
# AGE                       0.43   :   KINDNESS                  0.28
# being                     0.23   :   AGE                       0.25
# KINDNESS                  0.22   :   health                    0.25
# connection                0.22   :   abstemiousness            0.24
# commonness                0.20   :   fertility                 0.22
#
# EARLIER_WORK                     :   EARLY_STAGE                   (add)
# being                     0.24   :   timing                    0.27
# difficulty                0.21   :   age                       0.26
# good                      0.21   :   maturity                  0.24
# seniority                 0.20   :   height                    0.22
# thoughtfulness            0.20   :   potential                 0.21
#
# GENERAL_PRINCIPLE                :   BASIC_RULE                    (add)
# FAIRNESS                  0.40   :   MORALITY                  0.36
# MORALITY                  0.36   :   MANDATE                   0.36
# NECESSITY                 0.36   :   FAIRNESS                  0.35
# MANDATE                   0.33   :   dispensability            0.35
# majority                  0.32   :   NECESSITY                 0.31
#
# CERTAIN_CIRCUMSTANCE             :   PARTICULAR_CASE               (add)
# timing                    0.37   :   substantiality            0.36
# nature                    0.35   :   typicality                0.36
# COMMONNESS                0.32   :   seriousness               0.36
# finality                  0.32   :   fairness                  0.34
# corruptness               0.31   :   COMMONNESS                0.34
#
# VAST_AMOUNT                      :   HIGH_PRICE                    (add)
# QUANTITY                  0.50   :   price                     0.82
# size                      0.42   :   quality                   0.34
# complexity                0.35   :   volume                    0.30
# substantiality            0.33   :   QUANTITY                  0.29
# nature                    0.33   :   temperature               0.28
#
# IMPORTANT_PART                   :   SIGNIFICANT_ROLE              (add)
# IMPORTANCE                0.49   :   SIGNIFICANCE              0.41
# SIGNIFICANCE              0.42   :   IMPORTANCE                0.40
# good                      0.37   :   position                  0.39
# necessity                 0.37   :   RESPONSIBILITY            0.39
# RESPONSIBILITY            0.34   :   potential                 0.36
#
# NEW_LAW                          :   MODERN_LANGUAGE               (add)
# mandate                   0.36   :   modernity                 0.45
# legality                  0.30   :   abstractness              0.43
# lawfulness                0.29   :   foreignness               0.41
# morality                  0.26   :   otherness                 0.35
# measure                   0.26   :   prolixity                 0.34
#
# LARGE_NUMBER                     :   VAST_AMOUNT                   (add)
# SIZE                      0.50   :   QUANTITY                  0.50
# majority                  0.40   :   SIZE                      0.42
# QUANTITY                  0.38   :   complexity                0.35
# volume                    0.34   :   substantiality            0.33
# potential                 0.31   :   nature                    0.33
#
# OLD_PERSON                       :   ELDERLY_LADY                  (add)
# AGE                       0.43   :   KINDNESS                  0.28
# being                     0.23   :   AGE                       0.25
# KINDNESS                  0.22   :   health                    0.25
# connection                0.22   :   abstemiousness            0.24
# commonness                0.20   :   fertility                 0.22
#
# NEW_LIFE                         :   EARLY_AGE                     (add)
# happiness                 0.34   :   age                       0.84
# humanness                 0.34   :   maturity                  0.36
# domesticity               0.32   :   height                    0.35
# freedom                   0.31   :   dormancy                  0.26
# reality                   0.30   :   susceptibility            0.25
#
# POLITICAL_ACTION                 :   ECONOMIC_DEVELOPMENT          (add)
# action                    0.70   :   commerce                  0.41
# cowardice                 0.32   :   CRISIS                    0.31
# CRISIS                    0.32   :   health                    0.29
# morality                  0.31   :   potential                 0.27
# civility                  0.31   :   modernity                 0.25
#
# LARGE_NUMBER                     :   GREAT_MAJORITY                (add)
# size                      0.50   :   MAJORITY                  0.73
# MAJORITY                  0.40   :   good                      0.59
# quantity                  0.38   :   minority                  0.45
# volume                    0.34   :   pride                     0.33
# potential                 0.31   :   continuity                0.29
#
# CERTAIN_CIRCUMSTANCE             :   PARTICULAR_CASE               (add)
# timing                    0.37   :   substantiality            0.36
# nature                    0.35   :   typicality                0.36
# COMMONNESS                0.32   :   seriousness               0.36
# finality                  0.32   :   fairness                  0.34
# corruptness               0.31   :   COMMONNESS                0.34
#
# CERTAIN_CIRCUMSTANCE             :   PARTICULAR_CASE               (add)
# timing                    0.37   :   substantiality            0.36
# nature                    0.35   :   typicality                0.36
# COMMONNESS                0.32   :   seriousness               0.36
# finality                  0.32   :   fairness                  0.34
# corruptness               0.31   :   COMMONNESS                0.34
#
# IMPORTANT_PART                   :   SIGNIFICANT_ROLE              (add)
# IMPORTANCE                0.49   :   SIGNIFICANCE              0.41
# SIGNIFICANCE              0.42   :   IMPORTANCE                0.40
# good                      0.37   :   position                  0.39
# necessity                 0.37   :   RESPONSIBILITY            0.39
# RESPONSIBILITY            0.34   :   potential                 0.36
#
# DIFFERENT_KIND                   :   VARIOUS_FORM                  (add)
# good                      0.44   :   shape                     0.35
# commonality               0.36   :   regularity                0.28
# similarity                0.34   :   nature                    0.27
# sameness                  0.34   :   consistency               0.26
# ABSTRACTNESS              0.33   :   ABSTRACTNESS              0.26
#
# EFFECTIVE_WAY                    :   EFFICIENT_USE                 (add)
# EFFECTIVENESS             0.44   :   EFFECTIVENESS             0.34
# good                      0.36   :   capability                0.31
# ability                   0.29   :   tractability              0.29
# NECESSITY                 0.28   :   utility                   0.29
# difference                0.27   :   NECESSITY                 0.28
#
# OLDER_MAN                        :   ELDERLY_WOMAN                 (add)
# AGE                       0.53   :   AGE                       0.31
# ancestry                  0.25   :   kindness                  0.28
# SEX                       0.25   :   SEX                       0.27
# commonness                0.23   :   fertility                 0.27
# honesty                   0.22   :   health                    0.24
#
# FEDERAL_ASSEMBLY                 :   NATIONAL_GOVERNMENT           (add)
# MANDATE                   0.28   :   MANDATE                   0.36
# majority                  0.25   :   health                    0.31
# seniority                 0.24   :   independence              0.30
# MINORITY                  0.22   :   crisis                    0.28
# function                  0.22   :   MINORITY                  0.28
#
# LARGE_QUANTITY                   :   GREAT_MAJORITY                (add)
# quantity                  0.83   :   majority                  0.73
# size                      0.51   :   good                      0.59
# volume                    0.37   :   minority                  0.45
# quality                   0.36   :   pride                     0.33
# substantiality            0.35   :   continuity                0.29
#
# VAST_AMOUNT                      :   LARGE_QUANTITY                (add)
# QUANTITY                  0.50   :   QUANTITY                  0.83
# SIZE                      0.42   :   SIZE                      0.51
# complexity                0.35   :   volume                    0.37
# SUBSTANTIALITY            0.33   :   quality                   0.36
# nature                    0.33   :   SUBSTANTIALITY            0.35
#
# SPECIAL_CIRCUMSTANCE             :   PARTICULAR_CASE               (add)
# timing                    0.29   :   substantiality            0.36
# significance              0.28   :   typicality                0.36
# nature                    0.27   :   seriousness               0.36
# magnitude                 0.26   :   fairness                  0.34
# auspiciousness            0.26   :   commonness                0.34
#
# LOCAL_OFFICE                     :   NEW_TECHNOLOGY                (add)
# health                    0.23   :   capability                0.40
# COMMERCE                  0.22   :   convenience               0.29
# mandate                   0.22   :   potential                 0.29
# responsibility            0.19   :   COMMERCE                  0.28
# function                  0.19   :   creativity                0.28
#
# CERTAIN_CIRCUMSTANCE             :   PARTICULAR_CASE               (add)
# timing                    0.37   :   substantiality            0.36
# nature                    0.35   :   typicality                0.36
# COMMONNESS                0.32   :   seriousness               0.36
# finality                  0.32   :   fairness                  0.34
# corruptness               0.31   :   COMMONNESS                0.34
#
# VAST_AMOUNT                      :   HIGH_PRICE                    (add)
# QUANTITY                  0.50   :   price                     0.82
# size                      0.42   :   quality                   0.34
# complexity                0.35   :   volume                    0.30
# substantiality            0.33   :   QUANTITY                  0.29
# nature                    0.33   :   temperature               0.28
#
# IMPORTANT_PART                   :   SIGNIFICANT_ROLE              (add)
# IMPORTANCE                0.49   :   SIGNIFICANCE              0.41
# SIGNIFICANCE              0.42   :   IMPORTANCE                0.40
# good                      0.37   :   position                  0.39
# necessity                 0.37   :   RESPONSIBILITY            0.39
# RESPONSIBILITY            0.34   :   potential                 0.36
#
# EARLIER_WORK                     :   EARLY_STAGE                   (add)
# being                     0.24   :   timing                    0.27
# difficulty                0.21   :   age                       0.26
# good                      0.21   :   maturity                  0.24
# seniority                 0.20   :   height                    0.22
# thoughtfulness            0.20   :   potential                 0.21
#
# GENERAL_PRINCIPLE                :   BASIC_RULE                    (add)
# FAIRNESS                  0.40   :   MORALITY                  0.36
# MORALITY                  0.36   :   MANDATE                   0.36
# NECESSITY                 0.36   :   FAIRNESS                  0.35
# MANDATE                   0.33   :   dispensability            0.35
# majority                  0.32   :   NECESSITY                 0.31
#
# LARGE_QUANTITY                   :   GREAT_MAJORITY                (add)
# quantity                  0.83   :   majority                  0.73
# size                      0.51   :   good                      0.59
# volume                    0.37   :   minority                  0.45
# quality                   0.36   :   pride                     0.33
# substantiality            0.35   :   continuity                0.29
#
# EFFECTIVE_WAY                    :   EFFICIENT_USE                 (add)
# EFFECTIVENESS             0.44   :   EFFECTIVENESS             0.34
# good                      0.36   :   capability                0.31
# ability                   0.29   :   tractability              0.29
# NECESSITY                 0.28   :   utility                   0.29
# difference                0.27   :   NECESSITY                 0.28
#
# OLDER_MAN                        :   ELDERLY_WOMAN                 (add)
# AGE                       0.53   :   AGE                       0.31
# ancestry                  0.25   :   kindness                  0.28
# SEX                       0.25   :   SEX                       0.27
# commonness                0.23   :   fertility                 0.27
# honesty                   0.22   :   health                    0.24
#
# FEDERAL_ASSEMBLY                 :   NATIONAL_GOVERNMENT           (add)
# MANDATE                   0.28   :   MANDATE                   0.36
# majority                  0.25   :   health                    0.31
# seniority                 0.24   :   independence              0.30
# MINORITY                  0.22   :   crisis                    0.28
# function                  0.22   :   MINORITY                  0.28
#
# LARGE_NUMBER                     :   VAST_AMOUNT                   (add)
# SIZE                      0.50   :   QUANTITY                  0.50
# majority                  0.40   :   SIZE                      0.42
# QUANTITY                  0.38   :   complexity                0.35
# volume                    0.34   :   substantiality            0.33
# potential                 0.31   :   nature                    0.33
#
# OLDER_MAN                        :   ELDERLY_WOMAN                 (add)
# AGE                       0.53   :   AGE                       0.31
# ancestry                  0.25   :   kindness                  0.28
# SEX                       0.25   :   SEX                       0.27
# commonness                0.23   :   fertility                 0.27
# honesty                   0.22   :   health                    0.24
#
# FEDERAL_ASSEMBLY                 :   NATIONAL_GOVERNMENT           (add)
# MANDATE                   0.28   :   MANDATE                   0.36
# majority                  0.25   :   health                    0.31
# seniority                 0.24   :   independence              0.30
# MINORITY                  0.22   :   crisis                    0.28
# function                  0.22   :   MINORITY                  0.28
#
# LARGE_NUMBER                     :   GREAT_MAJORITY                (add)
# size                      0.50   :   MAJORITY                  0.73
# MAJORITY                  0.40   :   good                      0.59
# quantity                  0.38   :   minority                  0.45
# volume                    0.34   :   pride                     0.33
# potential                 0.31   :   continuity                0.29
#
# CERTAIN_CIRCUMSTANCE             :   PARTICULAR_CASE               (add)
# timing                    0.37   :   substantiality            0.36
# nature                    0.35   :   typicality                0.36
# COMMONNESS                0.32   :   seriousness               0.36
# finality                  0.32   :   fairness                  0.34
# corruptness               0.31   :   COMMONNESS                0.34
#
# IMPORTANT_PART                   :   SIGNIFICANT_ROLE              (add)
# IMPORTANCE                0.49   :   SIGNIFICANCE              0.41
# SIGNIFICANCE              0.42   :   IMPORTANCE                0.40
# good                      0.37   :   position                  0.39
# necessity                 0.37   :   RESPONSIBILITY            0.39
# RESPONSIBILITY            0.34   :   potential                 0.36
#
# LARGE_NUMBER                     :   VAST_AMOUNT                   (add)
# SIZE                      0.50   :   QUANTITY                  0.50
# majority                  0.40   :   SIZE                      0.42
# QUANTITY                  0.38   :   complexity                0.35
# volume                    0.34   :   substantiality            0.33
# potential                 0.31   :   nature                    0.33
#
# DIFFERENT_KIND                   :   VARIOUS_FORM                  (add)
# good                      0.44   :   shape                     0.35
# commonality               0.36   :   regularity                0.28
# similarity                0.34   :   nature                    0.27
# sameness                  0.34   :   consistency               0.26
# ABSTRACTNESS              0.33   :   ABSTRACTNESS              0.26
#
# OLDER_MAN                        :   ELDERLY_WOMAN                 (add)
# AGE                       0.53   :   AGE                       0.31
# ancestry                  0.25   :   kindness                  0.28
# SEX                       0.25   :   SEX                       0.27
# commonness                0.23   :   fertility                 0.27
# honesty                   0.22   :   health                    0.24
#
# IMPORTANT_PART                   :   SIGNIFICANT_ROLE              (add)
# IMPORTANCE                0.49   :   SIGNIFICANCE              0.41
# SIGNIFICANCE              0.42   :   IMPORTANCE                0.40
# good                      0.37   :   position                  0.39
# necessity                 0.37   :   RESPONSIBILITY            0.39
# RESPONSIBILITY            0.34   :   potential                 0.36
#
# OLD_PERSON                       :   ELDERLY_LADY                  (add)
# AGE                       0.43   :   KINDNESS                  0.28
# being                     0.23   :   AGE                       0.25
# KINDNESS                  0.22   :   health                    0.25
# connection                0.22   :   abstemiousness            0.24
# commonness                0.20   :   fertility                 0.22
#
# EFFECTIVE_WAY                    :   EFFICIENT_USE                 (add)
# EFFECTIVENESS             0.44   :   EFFECTIVENESS             0.34
# good                      0.36   :   capability                0.31
# ability                   0.29   :   tractability              0.29
# NECESSITY                 0.28   :   utility                   0.29
# difference                0.27   :   NECESSITY                 0.28
#
# LARGE_NUMBER                     :   GREAT_MAJORITY                (add)
# size                      0.50   :   MAJORITY                  0.73
# MAJORITY                  0.40   :   good                      0.59
# quantity                  0.38   :   minority                  0.45
# volume                    0.34   :   pride                     0.33
# potential                 0.31   :   continuity                0.29
#
# CERTAIN_CIRCUMSTANCE             :   PARTICULAR_CASE               (add)
# timing                    0.37   :   substantiality            0.36
# nature                    0.35   :   typicality                0.36
# COMMONNESS                0.32   :   seriousness               0.36
# finality                  0.32   :   fairness                  0.34
# corruptness               0.31   :   COMMONNESS                0.34
#
# NEW_INFORMATION                  :   FURTHER_EVIDENCE              (add)
# reassurance               0.28   :   likelihood                0.37
# intelligence              0.27   :   possibility               0.36
# convenience               0.26   :   credibility               0.33
# excitement                0.26   :   substantiality            0.31
# clarity                   0.25   :   truth                     0.30
#
# VAST_AMOUNT                      :   LARGE_QUANTITY                (add)
# QUANTITY                  0.50   :   QUANTITY                  0.83
# SIZE                      0.42   :   SIZE                      0.51
# complexity                0.35   :   volume                    0.37
# SUBSTANTIALITY            0.33   :   quality                   0.36
# nature                    0.33   :   SUBSTANTIALITY            0.35
#
# SPECIAL_CIRCUMSTANCE             :   PARTICULAR_CASE               (add)
# timing                    0.29   :   substantiality            0.36
# significance              0.28   :   typicality                0.36
# nature                    0.27   :   seriousness               0.36
# magnitude                 0.26   :   fairness                  0.34
# auspiciousness            0.26   :   commonness                0.34
#
# GOOD_PLACE                       :   HIGH_POINT                    (add)
# GOOD                      0.77   :   height                    0.31
# shape                     0.38   :   GOOD                      0.30
# position                  0.31   :   maturity                  0.25
# consistency               0.31   :   degree                    0.25
# pride                     0.30   :   speed                     0.24
#
# CERTAIN_CIRCUMSTANCE             :   PARTICULAR_CASE               (add)
# timing                    0.37   :   substantiality            0.36
# nature                    0.35   :   typicality                0.36
# COMMONNESS                0.32   :   seriousness               0.36
# finality                  0.32   :   fairness                  0.34
# corruptness               0.31   :   COMMONNESS                0.34
#
# VAST_AMOUNT                      :   HIGH_PRICE                    (add)
# QUANTITY                  0.50   :   price                     0.82
# size                      0.42   :   quality                   0.34
# complexity                0.35   :   volume                    0.30
# substantiality            0.33   :   QUANTITY                  0.29
# nature                    0.33   :   temperature               0.28
#
# NEW_LIFE                         :   ECONOMIC_DEVELOPMENT          (add)
# happiness                 0.34   :   commerce                  0.41
# humanness                 0.34   :   crisis                    0.31
# domesticity               0.32   :   health                    0.29
# freedom                   0.31   :   potential                 0.27
# reality                   0.30   :   modernity                 0.25
#
# LARGE_NUMBER                     :   VAST_AMOUNT                   (add)
# SIZE                      0.50   :   QUANTITY                  0.50
# majority                  0.40   :   SIZE                      0.42
# QUANTITY                  0.38   :   complexity                0.35
# volume                    0.34   :   substantiality            0.33
# potential                 0.31   :   nature                    0.33
#
# DIFFERENT_KIND                   :   VARIOUS_FORM                  (add)
# good                      0.44   :   shape                     0.35
# commonality               0.36   :   regularity                0.28
# similarity                0.34   :   nature                    0.27
# sameness                  0.34   :   consistency               0.26
# ABSTRACTNESS              0.33   :   ABSTRACTNESS              0.26
#
# FEDERAL_ASSEMBLY                 :   NATIONAL_GOVERNMENT           (add)
# MANDATE                   0.28   :   MANDATE                   0.36
# majority                  0.25   :   health                    0.31
# seniority                 0.24   :   independence              0.30
# MINORITY                  0.22   :   crisis                    0.28
# function                  0.22   :   MINORITY                  0.28
#
# BLACK_HAIR                       :   DARK_EYE                      (add)
# COMPLEXION                0.39   :   light                     0.43
# auspiciousness            0.30   :   mind                      0.37
# texture                   0.29   :   luminosity                0.33
# virginity                 0.29   :   COMPLEXION                0.30
# ancestry                  0.27   :   sharpness                 0.30
#
# GOOD_PLACE                       :   HIGH_POINT                    (add)
# GOOD                      0.77   :   height                    0.31
# shape                     0.38   :   GOOD                      0.30
# position                  0.31   :   maturity                  0.25
# consistency               0.31   :   degree                    0.25
# pride                     0.30   :   speed                     0.24
#
# NEW_INFORMATION                  :   FURTHER_EVIDENCE              (add)
# reassurance               0.28   :   likelihood                0.37
# intelligence              0.27   :   possibility               0.36
# convenience               0.26   :   credibility               0.33
# excitement                0.26   :   substantiality            0.31
# clarity                   0.25   :   truth                     0.30
#
# EFFECTIVE_WAY                    :   IMPORTANT_PART                (add)
# effectiveness             0.44   :   importance                0.49
# GOOD                      0.36   :   significance              0.42
# ability                   0.29   :   GOOD                      0.37
# NECESSITY                 0.28   :   NECESSITY                 0.37
# difference                0.27   :   responsibility            0.34
#
# BETTER_JOB                       :   GOOD_EFFECT                   (add)
# GOOD                      0.56   :   GOOD                      0.67
# position                  0.38   :   difference                0.41
# SHAPE                     0.29   :   SHAPE                     0.37
# consistency               0.29   :   effectiveness             0.32
# responsibility            0.29   :   strength                  0.30
#
# NEW_SITUATION                    :   DIFFERENT_KIND                (add)
# crisis                    0.42   :   good                      0.44
# position                  0.39   :   commonality               0.36
# possibility               0.34   :   similarity                0.34
# reality                   0.34   :   sameness                  0.34
# shape                     0.30   :   abstractness              0.33
#
# ECONOMIC_DEVELOPMENT             :   RURAL_COMMUNITY               (add)
# COMMERCE                  0.41   :   literacy                  0.39
# crisis                    0.31   :   minority                  0.32
# HEALTH                    0.29   :   HEALTH                    0.31
# potential                 0.27   :   COMMERCE                  0.27
# modernity                 0.25   :   generosity                0.27
#
# GENERAL_LEVEL                    :   FEDERAL_ASSEMBLY              (add)
# depth                     0.29   :   mandate                   0.28
# activeness                0.26   :   majority                  0.25
# magnitude                 0.26   :   seniority                 0.24
# price                     0.26   :   minority                  0.22
# maturity                  0.26   :   function                  0.22
#
# GENERAL_PRINCIPLE                :   PRESENT_POSITION              (add)
# fairness                  0.40   :   position                  0.80
# morality                  0.36   :   presence                  0.34
# necessity                 0.36   :   status                    0.33
# mandate                   0.33   :   stature                   0.32
# majority                  0.32   :   shape                     0.31
#
# AMERICAN_COUNTRY                 :   EUROPEAN_STATE                (add)
# CORRUPTNESS               0.30   :   typicality                0.22
# freedom                   0.23   :   function                  0.19
# evil                      0.23   :   essentiality              0.19
# activeness                0.21   :   dispensability            0.19
# repute                    0.20   :   CORRUPTNESS               0.19
#
# EARLY_STAGE                      :   LONG_PERIOD                   (add)
# timing                    0.27   :   duration                  0.48
# age                       0.26   :   dormancy                  0.32
# MATURITY                  0.24   :   length                    0.32
# height                    0.22   :   distance                  0.30
# potential                 0.21   :   MATURITY                  0.28
#
# CENTRAL_AUTHORITY                :   POLITICAL_ACTION              (add)
# mandate                   0.41   :   action                    0.70
# power                     0.38   :   cowardice                 0.32
# responsibility            0.35   :   crisis                    0.32
# legality                  0.31   :   morality                  0.31
# function                  0.29   :   civility                  0.31
#
# EARLY_EVENING                    :   PREVIOUS_DAY                  (add)
# TIMING                    0.19   :   duration                  0.34
# formality                 0.19   :   volume                    0.25
# action                    0.17   :   TIMING                    0.25
# light                     0.16   :   mind                      0.23
# good                      0.16   :   length                    0.22
#
# SPECIAL_CIRCUMSTANCE             :   PARTICULAR_CASE               (add)
# timing                    0.29   :   substantiality            0.36
# significance              0.28   :   typicality                0.36
# nature                    0.27   :   seriousness               0.36
# magnitude                 0.26   :   fairness                  0.34
# auspiciousness            0.26   :   commonness                0.34
#
# EFFICIENT_USE                    :   SIGNIFICANT_ROLE              (add)
# effectiveness             0.34   :   significance              0.41
# capability                0.31   :   importance                0.40
# tractability              0.29   :   position                  0.39
# utility                   0.29   :   responsibility            0.39
# necessity                 0.28   :   potential                 0.36
#
# ECONOMIC_CONDITION               :   AMERICAN_COUNTRY              (add)
# health                    0.42   :   corruptness               0.30
# crisis                    0.35   :   freedom                   0.23
# shape                     0.32   :   evil                      0.23
# cyclicity                 0.29   :   activeness                0.21
# commerce                  0.27   :   repute                    0.20
#
# EFFECTIVE_WAY                    :   PRACTICAL_DIFFICULTY          (add)
# effectiveness             0.44   :   difficulty                0.80
# good                      0.36   :   NECESSITY                 0.46
# ability                   0.29   :   practicality              0.45
# NECESSITY                 0.28   :   complexity                0.44
# difference                0.27   :   importance                0.42
#
# EFFICIENT_USE                    :   LITTLE_ROOM                   (add)
# effectiveness             0.34   :   comfort                   0.33
# capability                0.31   :   good                      0.29
# tractability              0.29   :   mind                      0.26
# utility                   0.29   :   actuality                 0.25
# necessity                 0.28   :   light                     0.24
#
# OLD_PERSON                       :   ELDERLY_LADY                  (add)
# AGE                       0.43   :   KINDNESS                  0.28
# being                     0.23   :   AGE                       0.25
# KINDNESS                  0.22   :   health                    0.25
# connection                0.22   :   abstemiousness            0.24
# commonness                0.20   :   fertility                 0.22
#
# EARLIER_WORK                     :   EARLY_STAGE                   (add)
# being                     0.24   :   timing                    0.27
# difficulty                0.21   :   age                       0.26
# good                      0.21   :   maturity                  0.24
# seniority                 0.20   :   height                    0.22
# thoughtfulness            0.20   :   potential                 0.21
#
# BETTER_JOB                       :   GOOD_PLACE                    (add)
# GOOD                      0.56   :   GOOD                      0.77
# POSITION                  0.38   :   SHAPE                     0.38
# SHAPE                     0.29   :   POSITION                  0.31
# CONSISTENCY               0.29   :   CONSISTENCY               0.31
# responsibility            0.29   :   pride                     0.30
#
# DARK_EYE                         :   LEFT_ARM                      (add)
# light                     0.43   :   pitch                     0.27
# mind                      0.37   :   position                  0.26
# luminosity                0.33   :   strength                  0.25
# complexion                0.30   :   length                    0.23
# sharpness                 0.30   :   direction                 0.23
#
# DIFFERENT_KIND                   :   VARIOUS_FORM                  (add)
# good                      0.44   :   shape                     0.35
# commonality               0.36   :   regularity                0.28
# similarity                0.34   :   nature                    0.27
# sameness                  0.34   :   consistency               0.26
# ABSTRACTNESS              0.33   :   ABSTRACTNESS              0.26
#
# LARGE_QUANTITY                   :   GREAT_MAJORITY                (add)
# quantity                  0.83   :   majority                  0.73
# size                      0.51   :   good                      0.59
# volume                    0.37   :   minority                  0.45
# quality                   0.36   :   pride                     0.33
# substantiality            0.35   :   continuity                0.29
#
# OLDER_MAN                        :   ELDERLY_WOMAN                 (add)
# AGE                       0.53   :   AGE                       0.31
# ancestry                  0.25   :   kindness                  0.28
# SEX                       0.25   :   SEX                       0.27
# commonness                0.23   :   fertility                 0.27
# honesty                   0.22   :   health                    0.24
#
# ECONOMIC_CONDITION               :   AMERICAN_COUNTRY              (add)
# health                    0.42   :   corruptness               0.30
# crisis                    0.35   :   freedom                   0.23
# shape                     0.32   :   evil                      0.23
# cyclicity                 0.29   :   activeness                0.21
# commerce                  0.27   :   repute                    0.20
#
# EARLIER_WORK                     :   EARLY_EVENING                 (add)
# being                     0.24   :   timing                    0.19
# difficulty                0.21   :   formality                 0.19
# GOOD                      0.21   :   action                    0.17
# seniority                 0.20   :   light                     0.16
# thoughtfulness            0.20   :   GOOD                      0.16
#
# PREVIOUS_DAY                     :   EARLY_AGE                     (add)
# duration                  0.34   :   age                       0.84
# volume                    0.25   :   maturity                  0.36
# timing                    0.25   :   height                    0.35
# mind                      0.23   :   dormancy                  0.26
# length                    0.22   :   susceptibility            0.25
#
# PUBLIC_BUILDING                  :   CENTRAL_AUTHORITY             (add)
# health                    0.29   :   mandate                   0.41
# civility                  0.24   :   power                     0.38
# integrity                 0.23   :   responsibility            0.35
# propriety                 0.21   :   legality                  0.31
# appropriateness           0.21   :   function                  0.29
#
# DIFFERENT_KIND                   :   VARIOUS_FORM                  (add)
# good                      0.44   :   shape                     0.35
# commonality               0.36   :   regularity                0.28
# similarity                0.34   :   nature                    0.27
# sameness                  0.34   :   consistency               0.26
# ABSTRACTNESS              0.33   :   ABSTRACTNESS              0.26
#
# EFFECTIVE_WAY                    :   EFFICIENT_USE                 (add)
# EFFECTIVENESS             0.44   :   EFFECTIVENESS             0.34
# good                      0.36   :   capability                0.31
# ability                   0.29   :   tractability              0.29
# NECESSITY                 0.28   :   utility                   0.29
# difference                0.27   :   NECESSITY                 0.28
#
# NEW_LIFE                         :   EARLY_AGE                     (add)
# happiness                 0.34   :   age                       0.84
# humanness                 0.34   :   maturity                  0.36
# domesticity               0.32   :   height                    0.35
# freedom                   0.31   :   dormancy                  0.26
# reality                   0.30   :   susceptibility            0.25
#
# LARGE_QUANTITY                   :   GREAT_MAJORITY                (add)
# quantity                  0.83   :   majority                  0.73
# size                      0.51   :   good                      0.59
# volume                    0.37   :   minority                  0.45
# quality                   0.36   :   pride                     0.33
# substantiality            0.35   :   continuity                0.29
#
# LONG_PERIOD                      :   SHORT_TIME                    (add)
# DURATION                  0.48   :   DURATION                  0.45
# dormancy                  0.32   :   LENGTH                    0.41
# LENGTH                    0.32   :   DISTANCE                  0.39
# DISTANCE                  0.30   :   timing                    0.30
# maturity                  0.28   :   good                      0.25
#
# RIGHT_HAND                       :   LEFT_ARM                      (add)
# DIRECTION                 0.32   :   PITCH                     0.27
# good                      0.29   :   POSITION                  0.26
# PITCH                     0.29   :   strength                  0.25
# mind                      0.26   :   length                    0.23
# POSITION                  0.25   :   DIRECTION                 0.23
#
# EFFECTIVE_WAY                    :   PRACTICAL_DIFFICULTY          (add)
# effectiveness             0.44   :   difficulty                0.80
# good                      0.36   :   NECESSITY                 0.46
# ability                   0.29   :   practicality              0.45
# NECESSITY                 0.28   :   complexity                0.44
# difference                0.27   :   importance                0.42
#
# MAJOR_ISSUE                      :   SOCIAL_EVENT                  (add)
# possibility               0.33   :   sociality                 0.36
# crisis                    0.30   :   sociability               0.34
# significance              0.30   :   awareness                 0.30
# importance                0.30   :   equality                  0.29
# legality                  0.29   :   literacy                  0.27
#
# HOT_WEATHER                      :   COLD_AIR                      (add)
# TEMPERATURE               0.44   :   TEMPERATURE               0.46
# WETNESS                   0.41   :   WETNESS                   0.38
# good                      0.28   :   smell                     0.32
# LIGHT                     0.27   :   LIGHT                     0.26
# taste                     0.24   :   staleness                 0.24
#
# WHOLE_COUNTRY                    :   GENERAL_PRINCIPLE             (add)
# good                      0.31   :   fairness                  0.40
# truth                     0.28   :   morality                  0.36
# mind                      0.28   :   necessity                 0.36
# corruptness               0.28   :   mandate                   0.33
# pride                     0.27   :   majority                  0.32
#
# GENERAL_PRINCIPLE                :   BASIC_RULE                    (add)
# FAIRNESS                  0.40   :   MORALITY                  0.36
# MORALITY                  0.36   :   MANDATE                   0.36
# NECESSITY                 0.36   :   FAIRNESS                  0.35
# MANDATE                   0.33   :   dispensability            0.35
# majority                  0.32   :   NECESSITY                 0.31
#
# DIFFERENT_KIND                   :   VARIOUS_FORM                  (add)
# good                      0.44   :   shape                     0.35
# commonality               0.36   :   regularity                0.28
# similarity                0.34   :   nature                    0.27
# sameness                  0.34   :   consistency               0.26
# ABSTRACTNESS              0.33   :   ABSTRACTNESS              0.26
#
# LARGE_QUANTITY                   :   GREAT_MAJORITY                (add)
# quantity                  0.83   :   majority                  0.73
# size                      0.51   :   good                      0.59
# volume                    0.37   :   minority                  0.45
# quality                   0.36   :   pride                     0.33
# substantiality            0.35   :   continuity                0.29
#
# SOCIAL_ACTIVITY                  :   POLITICAL_ACTION              (add)
# sociability               0.46   :   action                    0.70
# sociality                 0.44   :   cowardice                 0.32
# activeness                0.36   :   crisis                    0.32
# permissiveness            0.35   :   morality                  0.31
# acquisitiveness           0.33   :   civility                  0.31
#
# PUBLIC_BUILDING                  :   CENTRAL_AUTHORITY             (add)
# health                    0.29   :   mandate                   0.41
# civility                  0.24   :   power                     0.38
# integrity                 0.23   :   responsibility            0.35
# propriety                 0.21   :   legality                  0.31
# appropriateness           0.21   :   function                  0.29
#
# NEW_LAW                          :   BASIC_RULE                    (add)
# MANDATE                   0.36   :   MORALITY                  0.36
# legality                  0.30   :   MANDATE                   0.36
# lawfulness                0.29   :   fairness                  0.35
# MORALITY                  0.26   :   dispensability            0.35
# measure                   0.26   :   necessity                 0.31
#
# EFFECTIVE_WAY                    :   EFFICIENT_USE                 (add)
# EFFECTIVENESS             0.44   :   EFFECTIVENESS             0.34
# good                      0.36   :   capability                0.31
# ability                   0.29   :   tractability              0.29
# NECESSITY                 0.28   :   utility                   0.29
# difference                0.27   :   NECESSITY                 0.28
#
# ECONOMIC_DEVELOPMENT             :   RURAL_COMMUNITY               (add)
# COMMERCE                  0.41   :   literacy                  0.39
# crisis                    0.31   :   minority                  0.32
# HEALTH                    0.29   :   HEALTH                    0.31
# potential                 0.27   :   COMMERCE                  0.27
# modernity                 0.25   :   generosity                0.27
#
# SOCIAL_ACTIVITY                  :   ECONOMIC_CONDITION            (add)
# sociability               0.46   :   health                    0.42
# sociality                 0.44   :   crisis                    0.35
# activeness                0.36   :   shape                     0.32
# permissiveness            0.35   :   cyclicity                 0.29
# acquisitiveness           0.33   :   commerce                  0.27
#
# GOOD_PLACE                       :   HIGH_POINT                    (add)
# GOOD                      0.77   :   height                    0.31
# shape                     0.38   :   GOOD                      0.30
# position                  0.31   :   maturity                  0.25
# consistency               0.31   :   degree                    0.25
# pride                     0.30   :   speed                     0.24
#
# EARLY_STAGE                      :   LONG_PERIOD                   (add)
# timing                    0.27   :   duration                  0.48
# age                       0.26   :   dormancy                  0.32
# MATURITY                  0.24   :   length                    0.32
# height                    0.22   :   distance                  0.30
# potential                 0.21   :   MATURITY                  0.28
#
# ECONOMIC_PROBLEM                 :   PRACTICAL_DIFFICULTY          (add)
# crisis                    0.55   :   DIFFICULTY                0.80
# DIFFICULTY                0.37   :   NECESSITY                 0.46
# NECESSITY                 0.34   :   practicality              0.45
# IMPORTANCE                0.30   :   complexity                0.44
# dispensability            0.30   :   IMPORTANCE                0.42
#
# EFFECTIVE_WAY                    :   IMPORTANT_PART                (add)
# effectiveness             0.44   :   importance                0.49
# GOOD                      0.36   :   significance              0.42
# ability                   0.29   :   GOOD                      0.37
# NECESSITY                 0.28   :   NECESSITY                 0.37
# difference                0.27   :   responsibility            0.34
#
# EARLY_EVENING                    :   PREVIOUS_DAY                  (add)
# TIMING                    0.19   :   duration                  0.34
# formality                 0.19   :   volume                    0.25
# action                    0.17   :   TIMING                    0.25
# light                     0.16   :   mind                      0.23
# good                      0.16   :   length                    0.22
#
# DIFFERENT_PART                   :   VARIOUS_FORM                  (add)
# commonality               0.36   :   SHAPE                     0.35
# difference                0.36   :   regularity                0.28
# SHAPE                     0.35   :   nature                    0.27
# good                      0.34   :   consistency               0.26
# similarity                0.32   :   abstractness              0.26
#
# NEW_SITUATION                    :   DIFFERENT_KIND                (add)
# crisis                    0.42   :   good                      0.44
# position                  0.39   :   commonality               0.36
# possibility               0.34   :   similarity                0.34
# reality                   0.34   :   sameness                  0.34
# shape                     0.30   :   abstractness              0.33
#
# EFFICIENT_USE                    :   SIGNIFICANT_ROLE              (add)
# effectiveness             0.34   :   significance              0.41
# capability                0.31   :   importance                0.40
# tractability              0.29   :   position                  0.39
# utility                   0.29   :   responsibility            0.39
# necessity                 0.28   :   potential                 0.36
#
# BLACK_HAIR                       :   DARK_EYE                      (add)
# COMPLEXION                0.39   :   light                     0.43
# auspiciousness            0.30   :   mind                      0.37
# texture                   0.29   :   luminosity                0.33
# virginity                 0.29   :   COMPLEXION                0.30
# ancestry                  0.27   :   sharpness                 0.30
#
# AMERICAN_COUNTRY                 :   EUROPEAN_STATE                (add)
# CORRUPTNESS               0.30   :   typicality                0.22
# freedom                   0.23   :   function                  0.19
# evil                      0.23   :   essentiality              0.19
# activeness                0.21   :   dispensability            0.19
# repute                    0.20   :   CORRUPTNESS               0.19
#
# SMALL_HOUSE                      :   LITTLE_ROOM                   (add)
# size                      0.35   :   comfort                   0.33
# majority                  0.25   :   good                      0.29
# quantity                  0.23   :   mind                      0.26
# minority                  0.23   :   actuality                 0.25
# ordinariness              0.19   :   light                     0.24
#
# NEW_BODY                         :   WHOLE_SYSTEM                  (add)
# weight                    0.31   :   capability                0.32
# shape                     0.26   :   integrity                 0.32
# mind                      0.26   :   complexity                0.29
# fullness                  0.25   :   FUNCTION                  0.29
# FUNCTION                  0.24   :   fairness                  0.27
#
# GENERAL_PRINCIPLE                :   PRESENT_POSITION              (add)
# fairness                  0.40   :   position                  0.80
# morality                  0.36   :   presence                  0.34
# necessity                 0.36   :   status                    0.33
# mandate                   0.33   :   stature                   0.32
# majority                  0.32   :   shape                     0.31
#
# EARLY_STAGE                      :   LONG_PERIOD                   (add)
# timing                    0.27   :   duration                  0.48
# age                       0.26   :   dormancy                  0.32
# MATURITY                  0.24   :   length                    0.32
# height                    0.22   :   distance                  0.30
# potential                 0.21   :   MATURITY                  0.28
#
# ECONOMIC_PROBLEM                 :   PRACTICAL_DIFFICULTY          (add)
# crisis                    0.55   :   DIFFICULTY                0.80
# DIFFICULTY                0.37   :   NECESSITY                 0.46
# NECESSITY                 0.34   :   practicality              0.45
# IMPORTANCE                0.30   :   complexity                0.44
# dispensability            0.30   :   IMPORTANCE                0.42
#
# EARLY_EVENING                    :   PREVIOUS_DAY                  (add)
# TIMING                    0.19   :   duration                  0.34
# formality                 0.19   :   volume                    0.25
# action                    0.17   :   TIMING                    0.25
# light                     0.16   :   mind                      0.23
# good                      0.16   :   length                    0.22
#
# NORTHERN_REGION                  :   INDUSTRIAL_AREA               (add)
# crisis                    0.20   :   commerce                  0.27
# cyclicity                 0.19   :   utility                   0.26
# normality                 0.19   :   cleanness                 0.24
# minority                  0.18   :   potential                 0.21
# presence                  0.18   :   action                    0.19
#
# CERTAIN_CIRCUMSTANCE             :   ECONOMIC_CONDITION            (add)
# timing                    0.37   :   health                    0.42
# nature                    0.35   :   crisis                    0.35
# commonness                0.32   :   shape                     0.32
# finality                  0.32   :   cyclicity                 0.29
# corruptness               0.31   :   commerce                  0.27
#
# HIGH_PRICE                       :   LOW_COST                      (add)
# PRICE                     0.82   :   PRICE                     0.47
# QUALITY                   0.34   :   QUALITY                   0.34
# VOLUME                    0.30   :   convenience               0.28
# quantity                  0.29   :   ease                      0.25
# temperature               0.28   :   VOLUME                    0.25
#
# DARK_EYE                         :   LEFT_ARM                      (add)
# light                     0.43   :   pitch                     0.27
# mind                      0.37   :   position                  0.26
# luminosity                0.33   :   strength                  0.25
# complexion                0.30   :   length                    0.23
# sharpness                 0.30   :   direction                 0.23
#
# NEW_LIFE                         :   EARLY_AGE                     (add)
# happiness                 0.34   :   age                       0.84
# humanness                 0.34   :   maturity                  0.36
# domesticity               0.32   :   height                    0.35
# freedom                   0.31   :   dormancy                  0.26
# reality                   0.30   :   susceptibility            0.25
#
# OLDER_MAN                        :   ELDERLY_WOMAN                 (add)
# AGE                       0.53   :   AGE                       0.31
# ancestry                  0.25   :   kindness                  0.28
# SEX                       0.25   :   SEX                       0.27
# commonness                0.23   :   fertility                 0.27
# honesty                   0.22   :   health                    0.24
#
# NEW_INFORMATION                  :   FURTHER_EVIDENCE              (add)
# reassurance               0.28   :   likelihood                0.37
# intelligence              0.27   :   possibility               0.36
# convenience               0.26   :   credibility               0.33
# excitement                0.26   :   substantiality            0.31
# clarity                   0.25   :   truth                     0.30
#
# EFFECTIVE_WAY                    :   IMPORTANT_PART                (add)
# effectiveness             0.44   :   importance                0.49
# GOOD                      0.36   :   significance              0.42
# ability                   0.29   :   GOOD                      0.37
# NECESSITY                 0.28   :   NECESSITY                 0.37
# difference                0.27   :   responsibility            0.34
#
# BETTER_JOB                       :   GOOD_EFFECT                   (add)
# GOOD                      0.56   :   GOOD                      0.67
# position                  0.38   :   difference                0.41
# SHAPE                     0.29   :   SHAPE                     0.37
# consistency               0.29   :   effectiveness             0.32
# responsibility            0.29   :   strength                  0.30
#
# GOOD_PLACE                       :   HIGH_POINT                    (add)
# GOOD                      0.77   :   height                    0.31
# shape                     0.38   :   GOOD                      0.30
# position                  0.31   :   maturity                  0.25
# consistency               0.31   :   degree                    0.25
# pride                     0.30   :   speed                     0.24
#
# GENERAL_PRINCIPLE                :   PRESENT_POSITION              (add)
# fairness                  0.40   :   position                  0.80
# morality                  0.36   :   presence                  0.34
# necessity                 0.36   :   status                    0.33
# mandate                   0.33   :   stature                   0.32
# majority                  0.32   :   shape                     0.31
#
# DIFFERENT_PART                   :   VARIOUS_FORM                  (add)
# commonality               0.36   :   SHAPE                     0.35
# difference                0.36   :   regularity                0.28
# SHAPE                     0.35   :   nature                    0.27
# good                      0.34   :   consistency               0.26
# similarity                0.32   :   abstractness              0.26
#
# NEW_SITUATION                    :   DIFFERENT_KIND                (add)
# crisis                    0.42   :   good                      0.44
# position                  0.39   :   commonality               0.36
# possibility               0.34   :   similarity                0.34
# reality                   0.34   :   sameness                  0.34
# shape                     0.30   :   abstractness              0.33
#
# GOOD_PLACE                       :   HIGH_POINT                    (add)
# GOOD                      0.77   :   height                    0.31
# shape                     0.38   :   GOOD                      0.30
# position                  0.31   :   maturity                  0.25
# consistency               0.31   :   degree                    0.25
# pride                     0.30   :   speed                     0.24
#
# NEW_INFORMATION                  :   FURTHER_EVIDENCE              (add)
# reassurance               0.28   :   likelihood                0.37
# intelligence              0.27   :   possibility               0.36
# convenience               0.26   :   credibility               0.33
# excitement                0.26   :   substantiality            0.31
# clarity                   0.25   :   truth                     0.30
#
# SPECIAL_CIRCUMSTANCE             :   PARTICULAR_CASE               (add)
# timing                    0.29   :   substantiality            0.36
# significance              0.28   :   typicality                0.36
# nature                    0.27   :   seriousness               0.36
# magnitude                 0.26   :   fairness                  0.34
# auspiciousness            0.26   :   commonness                0.34
#
# NEW_SITUATION                    :   DIFFERENT_KIND                (add)
# crisis                    0.42   :   good                      0.44
# position                  0.39   :   commonality               0.36
# possibility               0.34   :   similarity                0.34
# reality                   0.34   :   sameness                  0.34
# shape                     0.30   :   abstractness              0.33
#
# VAST_AMOUNT                      :   LARGE_QUANTITY                (add)
# QUANTITY                  0.50   :   QUANTITY                  0.83
# SIZE                      0.42   :   SIZE                      0.51
# complexity                0.35   :   volume                    0.37
# SUBSTANTIALITY            0.33   :   quality                   0.36
# nature                    0.33   :   SUBSTANTIALITY            0.35
#
# POLITICAL_ACTION                 :   ECONOMIC_DEVELOPMENT          (add)
# action                    0.70   :   commerce                  0.41
# cowardice                 0.32   :   CRISIS                    0.31
# CRISIS                    0.32   :   health                    0.29
# morality                  0.31   :   potential                 0.27
# civility                  0.31   :   modernity                 0.25
#
# NEW_LIFE                         :   EARLY_AGE                     (add)
# happiness                 0.34   :   age                       0.84
# humanness                 0.34   :   maturity                  0.36
# domesticity               0.32   :   height                    0.35
# freedom                   0.31   :   dormancy                  0.26
# reality                   0.30   :   susceptibility            0.25
#
# BETTER_JOB                       :   GOOD_PLACE                    (add)
# GOOD                      0.56   :   GOOD                      0.77
# POSITION                  0.38   :   SHAPE                     0.38
# SHAPE                     0.29   :   POSITION                  0.31
# CONSISTENCY               0.29   :   CONSISTENCY               0.31
# responsibility            0.29   :   pride                     0.30
#
# VAST_AMOUNT                      :   HIGH_PRICE                    (add)
# QUANTITY                  0.50   :   price                     0.82
# size                      0.42   :   quality                   0.34
# complexity                0.35   :   volume                    0.30
# substantiality            0.33   :   QUANTITY                  0.29
# nature                    0.33   :   temperature               0.28
#
# RIGHT_HAND                       :   LEFT_ARM                      (add)
# DIRECTION                 0.32   :   PITCH                     0.27
# good                      0.29   :   POSITION                  0.26
# PITCH                     0.29   :   strength                  0.25
# mind                      0.26   :   length                    0.23
# POSITION                  0.25   :   DIRECTION                 0.23
#
# NEW_LAW                          :   BASIC_RULE                    (add)
# MANDATE                   0.36   :   MORALITY                  0.36
# legality                  0.30   :   MANDATE                   0.36
# lawfulness                0.29   :   fairness                  0.35
# MORALITY                  0.26   :   dispensability            0.35
# measure                   0.26   :   necessity                 0.31
#
# DIFFERENT_PART                   :   NORTHERN_REGION               (add)
# commonality               0.36   :   crisis                    0.20
# difference                0.36   :   cyclicity                 0.19
# shape                     0.35   :   normality                 0.19
# good                      0.34   :   minority                  0.18
# similarity                0.32   :   presence                  0.18
#
# SIMILAR_RESULT                   :   GOOD_EFFECT                   (add)
# likelihood                0.35   :   good                      0.67
# similarity                0.33   :   difference                0.41
# timing                    0.26   :   shape                     0.37
# potential                 0.26   :   effectiveness             0.32
# connection                0.25   :   strength                  0.30
#
# LARGE_NUMBER                     :   GREAT_MAJORITY                (add)
# size                      0.50   :   MAJORITY                  0.73
# MAJORITY                  0.40   :   good                      0.59
# quantity                  0.38   :   minority                  0.45
# volume                    0.34   :   pride                     0.33
# potential                 0.31   :   continuity                0.29
#
# CERTAIN_CIRCUMSTANCE             :   PARTICULAR_CASE               (add)
# timing                    0.37   :   substantiality            0.36
# nature                    0.35   :   typicality                0.36
# COMMONNESS                0.32   :   seriousness               0.36
# finality                  0.32   :   fairness                  0.34
# corruptness               0.31   :   COMMONNESS                0.34
#
# LONG_PERIOD                      :   SHORT_TIME                    (add)
# DURATION                  0.48   :   DURATION                  0.45
# dormancy                  0.32   :   LENGTH                    0.41
# LENGTH                    0.32   :   DISTANCE                  0.39
# DISTANCE                  0.30   :   timing                    0.30
# maturity                  0.28   :   good                      0.25
#
# NEW_LAW                          :   BASIC_RULE                    (add)
# MANDATE                   0.36   :   MORALITY                  0.36
# legality                  0.30   :   MANDATE                   0.36
# lawfulness                0.29   :   fairness                  0.35
# MORALITY                  0.26   :   dispensability            0.35
# measure                   0.26   :   necessity                 0.31
#
# PREVIOUS_DAY                     :   EARLY_AGE                     (add)
# duration                  0.34   :   age                       0.84
# volume                    0.25   :   maturity                  0.36
# timing                    0.25   :   height                    0.35
# mind                      0.23   :   dormancy                  0.26
# length                    0.22   :   susceptibility            0.25
#
# HOT_WEATHER                      :   COLD_AIR                      (add)
# TEMPERATURE               0.44   :   TEMPERATURE               0.46
# WETNESS                   0.41   :   WETNESS                   0.38
# good                      0.28   :   smell                     0.32
# LIGHT                     0.27   :   LIGHT                     0.26
# taste                     0.24   :   staleness                 0.24
#
# MAJOR_ISSUE                      :   SOCIAL_EVENT                  (add)
# possibility               0.33   :   sociality                 0.36
# crisis                    0.30   :   sociability               0.34
# significance              0.30   :   awareness                 0.30
# importance                0.30   :   equality                  0.29
# legality                  0.29   :   literacy                  0.27
#
# DIFFERENT_PART                   :   NORTHERN_REGION               (add)
# commonality               0.36   :   crisis                    0.20
# difference                0.36   :   cyclicity                 0.19
# shape                     0.35   :   normality                 0.19
# good                      0.34   :   minority                  0.18
# similarity                0.32   :   presence                  0.18
#
# HIGH_PRICE                       :   LOW_COST                      (add)
# PRICE                     0.82   :   PRICE                     0.47
# QUALITY                   0.34   :   QUALITY                   0.34
# VOLUME                    0.30   :   convenience               0.28
# quantity                  0.29   :   ease                      0.25
# temperature               0.28   :   VOLUME                    0.25
#
# EARLIER_WORK                     :   EARLY_STAGE                   (add)
# being                     0.24   :   timing                    0.27
# difficulty                0.21   :   age                       0.26
# good                      0.21   :   maturity                  0.24
# seniority                 0.20   :   height                    0.22
# thoughtfulness            0.20   :   potential                 0.21
#
# OLDER_MAN                        :   ELDERLY_WOMAN                 (add)
# AGE                       0.53   :   AGE                       0.31
# ancestry                  0.25   :   kindness                  0.28
# SEX                       0.25   :   SEX                       0.27
# commonness                0.23   :   fertility                 0.27
# honesty                   0.22   :   health                    0.24
#
# FEDERAL_ASSEMBLY                 :   NATIONAL_GOVERNMENT           (add)
# MANDATE                   0.28   :   MANDATE                   0.36
# majority                  0.25   :   health                    0.31
# seniority                 0.24   :   independence              0.30
# MINORITY                  0.22   :   crisis                    0.28
# function                  0.22   :   MINORITY                  0.28
#
# OLD_PERSON                       :   ELDERLY_LADY                  (add)
# AGE                       0.43   :   KINDNESS                  0.28
# being                     0.23   :   AGE                       0.25
# KINDNESS                  0.22   :   health                    0.25
# connection                0.22   :   abstemiousness            0.24
# commonness                0.20   :   fertility                 0.22
#
# GENERAL_PRINCIPLE                :   BASIC_RULE                    (add)
# FAIRNESS                  0.40   :   MORALITY                  0.36
# MORALITY                  0.36   :   MANDATE                   0.36
# NECESSITY                 0.36   :   FAIRNESS                  0.35
# MANDATE                   0.33   :   dispensability            0.35
# majority                  0.32   :   NECESSITY                 0.31
#
# LARGE_QUANTITY                   :   GREAT_MAJORITY                (add)
# quantity                  0.83   :   majority                  0.73
# size                      0.51   :   good                      0.59
# volume                    0.37   :   minority                  0.45
# quality                   0.36   :   pride                     0.33
# substantiality            0.35   :   continuity                0.29
#
# EFFECTIVE_WAY                    :   EFFICIENT_USE                 (add)
# EFFECTIVENESS             0.44   :   EFFECTIVENESS             0.34
# good                      0.36   :   capability                0.31
# ability                   0.29   :   tractability              0.29
# NECESSITY                 0.28   :   utility                   0.29
# difference                0.27   :   NECESSITY                 0.28
#
# FEDERAL_ASSEMBLY                 :   NATIONAL_GOVERNMENT           (add)
# MANDATE                   0.28   :   MANDATE                   0.36
# majority                  0.25   :   health                    0.31
# seniority                 0.24   :   independence              0.30
# MINORITY                  0.22   :   crisis                    0.28
# function                  0.22   :   MINORITY                  0.28
#
# SPECIAL_CIRCUMSTANCE             :   PARTICULAR_CASE               (add)
# timing                    0.29   :   substantiality            0.36
# significance              0.28   :   typicality                0.36
# nature                    0.27   :   seriousness               0.36
# magnitude                 0.26   :   fairness                  0.34
# auspiciousness            0.26   :   commonness                0.34
#
# SMALL_HOUSE                      :   LITTLE_ROOM                   (add)
# size                      0.35   :   comfort                   0.33
# majority                  0.25   :   good                      0.29
# quantity                  0.23   :   mind                      0.26
# minority                  0.23   :   actuality                 0.25
# ordinariness              0.19   :   light                     0.24
#
# NEW_INFORMATION                  :   FURTHER_EVIDENCE              (add)
# reassurance               0.28   :   likelihood                0.37
# intelligence              0.27   :   possibility               0.36
# convenience               0.26   :   credibility               0.33
# excitement                0.26   :   substantiality            0.31
# clarity                   0.25   :   truth                     0.30
#
# SPECIAL_CIRCUMSTANCE             :   PARTICULAR_CASE               (add)
# timing                    0.29   :   substantiality            0.36
# significance              0.28   :   typicality                0.36
# nature                    0.27   :   seriousness               0.36
# magnitude                 0.26   :   fairness                  0.34
# auspiciousness            0.26   :   commonness                0.34
#
# ECONOMIC_PROBLEM                 :   PRACTICAL_DIFFICULTY          (add)
# crisis                    0.55   :   DIFFICULTY                0.80
# DIFFICULTY                0.37   :   NECESSITY                 0.46
# NECESSITY                 0.34   :   practicality              0.45
# IMPORTANCE                0.30   :   complexity                0.44
# dispensability            0.30   :   IMPORTANCE                0.42
#
# EFFECTIVE_WAY                    :   IMPORTANT_PART                (add)
# effectiveness             0.44   :   importance                0.49
# GOOD                      0.36   :   significance              0.42
# ability                   0.29   :   GOOD                      0.37
# NECESSITY                 0.28   :   NECESSITY                 0.37
# difference                0.27   :   responsibility            0.34
#
# HIGH_PRICE                       :   SHORT_TIME                    (add)
# price                     0.82   :   duration                  0.45
# quality                   0.34   :   length                    0.41
# volume                    0.30   :   distance                  0.39
# quantity                  0.29   :   timing                    0.30
# temperature               0.28   :   good                      0.25
#
# NEW_SITUATION                    :   DIFFERENT_KIND                (add)
# crisis                    0.42   :   good                      0.44
# position                  0.39   :   commonality               0.36
# possibility               0.34   :   similarity                0.34
# reality                   0.34   :   sameness                  0.34
# shape                     0.30   :   abstractness              0.33
#
# GOOD_PLACE                       :   HIGH_POINT                    (add)
# GOOD                      0.77   :   height                    0.31
# shape                     0.38   :   GOOD                      0.30
# position                  0.31   :   maturity                  0.25
# consistency               0.31   :   degree                    0.25
# pride                     0.30   :   speed                     0.24
#
# NEW_SITUATION                    :   DIFFERENT_KIND                (add)
# crisis                    0.42   :   good                      0.44
# position                  0.39   :   commonality               0.36
# possibility               0.34   :   similarity                0.34
# reality                   0.34   :   sameness                  0.34
# shape                     0.30   :   abstractness              0.33
#
# CERTAIN_CIRCUMSTANCE             :   PARTICULAR_CASE               (add)
# timing                    0.37   :   substantiality            0.36
# nature                    0.35   :   typicality                0.36
# COMMONNESS                0.32   :   seriousness               0.36
# finality                  0.32   :   fairness                  0.34
# corruptness               0.31   :   COMMONNESS                0.34
#
# NEW_LAW                          :   BASIC_RULE                    (add)
# MANDATE                   0.36   :   MORALITY                  0.36
# legality                  0.30   :   MANDATE                   0.36
# lawfulness                0.29   :   fairness                  0.35
# MORALITY                  0.26   :   dispensability            0.35
# measure                   0.26   :   necessity                 0.31
#
# EFFECTIVE_WAY                    :   EFFICIENT_USE                 (add)
# EFFECTIVENESS             0.44   :   EFFECTIVENESS             0.34
# good                      0.36   :   capability                0.31
# ability                   0.29   :   tractability              0.29
# NECESSITY                 0.28   :   utility                   0.29
# difference                0.27   :   NECESSITY                 0.28
#
# OLDER_MAN                        :   ELDERLY_WOMAN                 (add)
# AGE                       0.53   :   AGE                       0.31
# ancestry                  0.25   :   kindness                  0.28
# SEX                       0.25   :   SEX                       0.27
# commonness                0.23   :   fertility                 0.27
# honesty                   0.22   :   health                    0.24
#
# FEDERAL_ASSEMBLY                 :   NATIONAL_GOVERNMENT           (add)
# MANDATE                   0.28   :   MANDATE                   0.36
# majority                  0.25   :   health                    0.31
# seniority                 0.24   :   independence              0.30
# MINORITY                  0.22   :   crisis                    0.28
# function                  0.22   :   MINORITY                  0.28
#
# SPECIAL_CIRCUMSTANCE             :   ELDERLY_LADY                  (add)
# timing                    0.29   :   kindness                  0.28
# significance              0.28   :   age                       0.25
# nature                    0.27   :   health                    0.25
# magnitude                 0.26   :   abstemiousness            0.24
# auspiciousness            0.26   :   fertility                 0.22
#
# CERTAIN_CIRCUMSTANCE             :   PARTICULAR_CASE               (add)
# timing                    0.37   :   substantiality            0.36
# nature                    0.35   :   typicality                0.36
# COMMONNESS                0.32   :   seriousness               0.36
# finality                  0.32   :   fairness                  0.34
# corruptness               0.31   :   COMMONNESS                0.34
#
# SOCIAL_ACTIVITY                  :   POLITICAL_ACTION              (add)
# sociability               0.46   :   action                    0.70
# sociality                 0.44   :   cowardice                 0.32
# activeness                0.36   :   crisis                    0.32
# permissiveness            0.35   :   morality                  0.31
# acquisitiveness           0.33   :   civility                  0.31
#
# IMPORTANT_PART                   :   SIGNIFICANT_ROLE              (add)
# IMPORTANCE                0.49   :   SIGNIFICANCE              0.41
# SIGNIFICANCE              0.42   :   IMPORTANCE                0.40
# good                      0.37   :   position                  0.39
# necessity                 0.37   :   RESPONSIBILITY            0.39
# RESPONSIBILITY            0.34   :   potential                 0.36
#
# CENTRAL_AUTHORITY                :   LOCAL_OFFICE                  (add)
# MANDATE                   0.41   :   health                    0.23
# power                     0.38   :   commerce                  0.22
# RESPONSIBILITY            0.35   :   MANDATE                   0.22
# legality                  0.31   :   RESPONSIBILITY            0.19
# FUNCTION                  0.29   :   FUNCTION                  0.19
#
# EFFECTIVE_WAY                    :   EFFICIENT_USE                 (add)
# EFFECTIVENESS             0.44   :   EFFECTIVENESS             0.34
# good                      0.36   :   capability                0.31
# ability                   0.29   :   tractability              0.29
# NECESSITY                 0.28   :   utility                   0.29
# difference                0.27   :   NECESSITY                 0.28
#
# SOCIAL_EVENT                     :   SPECIAL_CIRCUMSTANCE          (add)
# sociality                 0.36   :   timing                    0.29
# sociability               0.34   :   significance              0.28
# awareness                 0.30   :   nature                    0.27
# equality                  0.29   :   magnitude                 0.26
# literacy                  0.27   :   auspiciousness            0.26
#
# BETTER_JOB                       :   GOOD_PLACE                    (add)
# GOOD                      0.56   :   GOOD                      0.77
# POSITION                  0.38   :   SHAPE                     0.38
# SHAPE                     0.29   :   POSITION                  0.31
# CONSISTENCY               0.29   :   CONSISTENCY               0.31
# responsibility            0.29   :   pride                     0.30
#
# FEDERAL_ASSEMBLY                 :   NATIONAL_GOVERNMENT           (add)
# MANDATE                   0.28   :   MANDATE                   0.36
# majority                  0.25   :   health                    0.31
# seniority                 0.24   :   independence              0.30
# MINORITY                  0.22   :   crisis                    0.28
# function                  0.22   :   MINORITY                  0.28
#
# LARGE_QUANTITY                   :   GREAT_MAJORITY                (add)
# quantity                  0.83   :   majority                  0.73
# size                      0.51   :   good                      0.59
# volume                    0.37   :   minority                  0.45
# quality                   0.36   :   pride                     0.33
# substantiality            0.35   :   continuity                0.29
#
# SIMILAR_RESULT                   :   GOOD_EFFECT                   (add)
# likelihood                0.35   :   good                      0.67
# similarity                0.33   :   difference                0.41
# timing                    0.26   :   shape                     0.37
# potential                 0.26   :   effectiveness             0.32
# connection                0.25   :   strength                  0.30
#
# NEW_LAW                          :   BASIC_RULE                    (add)
# MANDATE                   0.36   :   MORALITY                  0.36
# legality                  0.30   :   MANDATE                   0.36
# lawfulness                0.29   :   fairness                  0.35
# MORALITY                  0.26   :   dispensability            0.35
# measure                   0.26   :   necessity                 0.31
#
# PUBLIC_BUILDING                  :   CENTRAL_AUTHORITY             (add)
# health                    0.29   :   mandate                   0.41
# civility                  0.24   :   power                     0.38
# integrity                 0.23   :   responsibility            0.35
# propriety                 0.21   :   legality                  0.31
# appropriateness           0.21   :   function                  0.29
#
# MAJOR_ISSUE                      :   SOCIAL_EVENT                  (add)
# possibility               0.33   :   sociality                 0.36
# crisis                    0.30   :   sociability               0.34
# significance              0.30   :   awareness                 0.30
# importance                0.30   :   equality                  0.29
# legality                  0.29   :   literacy                  0.27
#
# LARGE_NUMBER                     :   GREAT_MAJORITY                (add)
# size                      0.50   :   MAJORITY                  0.73
# MAJORITY                  0.40   :   good                      0.59
# quantity                  0.38   :   minority                  0.45
# volume                    0.34   :   pride                     0.33
# potential                 0.31   :   continuity                0.29
#
# SOCIAL_ACTIVITY                  :   POLITICAL_ACTION              (add)
# sociability               0.46   :   action                    0.70
# sociality                 0.44   :   cowardice                 0.32
# activeness                0.36   :   crisis                    0.32
# permissiveness            0.35   :   morality                  0.31
# acquisitiveness           0.33   :   civility                  0.31
#
# RIGHT_HAND                       :   LEFT_ARM                      (add)
# DIRECTION                 0.32   :   PITCH                     0.27
# good                      0.29   :   POSITION                  0.26
# PITCH                     0.29   :   strength                  0.25
# mind                      0.26   :   length                    0.23
# POSITION                  0.25   :   DIRECTION                 0.23
#
# IMPORTANT_PART                   :   SIGNIFICANT_ROLE              (add)
# IMPORTANCE                0.49   :   SIGNIFICANCE              0.41
# SIGNIFICANCE              0.42   :   IMPORTANCE                0.40
# good                      0.37   :   position                  0.39
# necessity                 0.37   :   RESPONSIBILITY            0.39
# RESPONSIBILITY            0.34   :   potential                 0.36
#
# NEW_SITUATION                    :   PRESENT_POSITION              (add)
# crisis                    0.42   :   POSITION                  0.80
# POSITION                  0.39   :   presence                  0.34
# possibility               0.34   :   status                    0.33
# reality                   0.34   :   stature                   0.32
# SHAPE                     0.30   :   SHAPE                     0.31
#
# OLD_PERSON                       :   ELDERLY_LADY                  (add)
# AGE                       0.43   :   KINDNESS                  0.28
# being                     0.23   :   AGE                       0.25
# KINDNESS                  0.22   :   health                    0.25
# connection                0.22   :   abstemiousness            0.24
# commonness                0.20   :   fertility                 0.22
#
# NEW_LIFE                         :   EARLY_AGE                     (add)
# happiness                 0.34   :   age                       0.84
# humanness                 0.34   :   maturity                  0.36
# domesticity               0.32   :   height                    0.35
# freedom                   0.31   :   dormancy                  0.26
# reality                   0.30   :   susceptibility            0.25
#
# OLDER_MAN                        :   ELDERLY_WOMAN                 (add)
# AGE                       0.53   :   AGE                       0.31
# ancestry                  0.25   :   kindness                  0.28
# SEX                       0.25   :   SEX                       0.27
# commonness                0.23   :   fertility                 0.27
# honesty                   0.22   :   health                    0.24
#
# EARLIER_WORK                     :   EARLY_STAGE                   (add)
# being                     0.24   :   timing                    0.27
# difficulty                0.21   :   age                       0.26
# good                      0.21   :   maturity                  0.24
# seniority                 0.20   :   height                    0.22
# thoughtfulness            0.20   :   potential                 0.21
#
# GOOD_PLACE                       :   HIGH_POINT                    (add)
# GOOD                      0.77   :   height                    0.31
# shape                     0.38   :   GOOD                      0.30
# position                  0.31   :   maturity                  0.25
# consistency               0.31   :   degree                    0.25
# pride                     0.30   :   speed                     0.24
#
# ECONOMIC_DEVELOPMENT             :   RURAL_COMMUNITY               (add)
# COMMERCE                  0.41   :   literacy                  0.39
# crisis                    0.31   :   minority                  0.32
# HEALTH                    0.29   :   HEALTH                    0.31
# potential                 0.27   :   COMMERCE                  0.27
# modernity                 0.25   :   generosity                0.27
#
# SMALL_HOUSE                      :   LITTLE_ROOM                   (add)
# size                      0.35   :   comfort                   0.33
# majority                  0.25   :   good                      0.29
# quantity                  0.23   :   mind                      0.26
# minority                  0.23   :   actuality                 0.25
# ordinariness              0.19   :   light                     0.24
#
# SIMILAR_RESULT                   :   BASIC_RULE                    (add)
# likelihood                0.35   :   morality                  0.36
# similarity                0.33   :   mandate                   0.36
# timing                    0.26   :   fairness                  0.35
# potential                 0.26   :   dispensability            0.35
# connection                0.25   :   necessity                 0.31
#
# ECONOMIC_PROBLEM                 :   PRACTICAL_DIFFICULTY          (add)
# crisis                    0.55   :   DIFFICULTY                0.80
# DIFFICULTY                0.37   :   NECESSITY                 0.46
# NECESSITY                 0.34   :   practicality              0.45
# IMPORTANCE                0.30   :   complexity                0.44
# dispensability            0.30   :   IMPORTANCE                0.42
#
# EFFECTIVE_WAY                    :   IMPORTANT_PART                (add)
# effectiveness             0.44   :   importance                0.49
# GOOD                      0.36   :   significance              0.42
# ability                   0.29   :   GOOD                      0.37
# NECESSITY                 0.28   :   NECESSITY                 0.37
# difference                0.27   :   responsibility            0.34
#
# DIFFERENT_PART                   :   VARIOUS_FORM                  (add)
# commonality               0.36   :   SHAPE                     0.35
# difference                0.36   :   regularity                0.28
# SHAPE                     0.35   :   nature                    0.27
# good                      0.34   :   consistency               0.26
# similarity                0.32   :   abstractness              0.26
#
# AMERICAN_COUNTRY                 :   EUROPEAN_STATE                (add)
# CORRUPTNESS               0.30   :   typicality                0.22
# freedom                   0.23   :   function                  0.19
# evil                      0.23   :   essentiality              0.19
# activeness                0.21   :   dispensability            0.19
# repute                    0.20   :   CORRUPTNESS               0.19
#
# SMALL_HOUSE                      :   LITTLE_ROOM                   (add)
# size                      0.35   :   comfort                   0.33
# majority                  0.25   :   good                      0.29
# quantity                  0.23   :   mind                      0.26
# minority                  0.23   :   actuality                 0.25
# ordinariness              0.19   :   light                     0.24
#
# EARLY_STAGE                      :   LONG_PERIOD                   (add)
# timing                    0.27   :   duration                  0.48
# age                       0.26   :   dormancy                  0.32
# MATURITY                  0.24   :   length                    0.32
# height                    0.22   :   distance                  0.30
# potential                 0.21   :   MATURITY                  0.28
#
# EFFECTIVE_WAY                    :   IMPORTANT_PART                (add)
# effectiveness             0.44   :   importance                0.49
# GOOD                      0.36   :   significance              0.42
# ability                   0.29   :   GOOD                      0.37
# NECESSITY                 0.28   :   NECESSITY                 0.37
# difference                0.27   :   responsibility            0.34
#
# SPECIAL_CIRCUMSTANCE             :   PARTICULAR_CASE               (add)
# timing                    0.29   :   substantiality            0.36
# significance              0.28   :   typicality                0.36
# nature                    0.27   :   seriousness               0.36
# magnitude                 0.26   :   fairness                  0.34
# auspiciousness            0.26   :   commonness                0.34
#
# NEW_INFORMATION                  :   FURTHER_EVIDENCE              (add)
# reassurance               0.28   :   likelihood                0.37
# intelligence              0.27   :   possibility               0.36
# convenience               0.26   :   credibility               0.33
# excitement                0.26   :   substantiality            0.31
# clarity                   0.25   :   truth                     0.30
#
# SMALL_HOUSE                      :   LITTLE_ROOM                   (add)
# size                      0.35   :   comfort                   0.33
# majority                  0.25   :   good                      0.29
# quantity                  0.23   :   mind                      0.26
# minority                  0.23   :   actuality                 0.25
# ordinariness              0.19   :   light                     0.24
#
# SOCIAL_ACTIVITY                  :   ECONOMIC_CONDITION            (add)
# sociability               0.46   :   health                    0.42
# sociality                 0.44   :   crisis                    0.35
# activeness                0.36   :   shape                     0.32
# permissiveness            0.35   :   cyclicity                 0.29
# acquisitiveness           0.33   :   commerce                  0.27
#
# EFFECTIVE_WAY                    :   IMPORTANT_PART                (add)
# effectiveness             0.44   :   importance                0.49
# GOOD                      0.36   :   significance              0.42
# ability                   0.29   :   GOOD                      0.37
# NECESSITY                 0.28   :   NECESSITY                 0.37
# difference                0.27   :   responsibility            0.34
#
# DIFFERENT_PART                   :   VARIOUS_FORM                  (add)
# commonality               0.36   :   SHAPE                     0.35
# difference                0.36   :   regularity                0.28
# SHAPE                     0.35   :   nature                    0.27
# good                      0.34   :   consistency               0.26
# similarity                0.32   :   abstractness              0.26
#
# LARGE_NUMBER                     :   GREAT_MAJORITY                (add)
# size                      0.50   :   MAJORITY                  0.73
# MAJORITY                  0.40   :   good                      0.59
# quantity                  0.38   :   minority                  0.45
# volume                    0.34   :   pride                     0.33
# potential                 0.31   :   continuity                0.29
#
# SOCIAL_ACTIVITY                  :   POLITICAL_ACTION              (add)
# sociability               0.46   :   action                    0.70
# sociality                 0.44   :   cowardice                 0.32
# activeness                0.36   :   crisis                    0.32
# permissiveness            0.35   :   morality                  0.31
# acquisitiveness           0.33   :   civility                  0.31
#
# WHOLE_COUNTRY                    :   GENERAL_PRINCIPLE             (add)
# good                      0.31   :   fairness                  0.40
# truth                     0.28   :   morality                  0.36
# mind                      0.28   :   necessity                 0.36
# corruptness               0.28   :   mandate                   0.33
# pride                     0.27   :   majority                  0.32
#
# CERTAIN_CIRCUMSTANCE             :   ECONOMIC_CONDITION            (add)
# timing                    0.37   :   health                    0.42
# nature                    0.35   :   crisis                    0.35
# commonness                0.32   :   shape                     0.32
# finality                  0.32   :   cyclicity                 0.29
# corruptness               0.31   :   commerce                  0.27
#
# POLITICAL_ACTION                 :   ECONOMIC_DEVELOPMENT          (add)
# action                    0.70   :   commerce                  0.41
# cowardice                 0.32   :   CRISIS                    0.31
# CRISIS                    0.32   :   health                    0.29
# morality                  0.31   :   potential                 0.27
# civility                  0.31   :   modernity                 0.25
#
# OLD_PERSON                       :   ELDERLY_LADY                  (add)
# AGE                       0.43   :   KINDNESS                  0.28
# being                     0.23   :   AGE                       0.25
# KINDNESS                  0.22   :   health                    0.25
# connection                0.22   :   abstemiousness            0.24
# commonness                0.20   :   fertility                 0.22
#
# EARLIER_WORK                     :   EARLY_STAGE                   (add)
# being                     0.24   :   timing                    0.27
# difficulty                0.21   :   age                       0.26
# good                      0.21   :   maturity                  0.24
# seniority                 0.20   :   height                    0.22
# thoughtfulness            0.20   :   potential                 0.21
#
# DIFFERENT_KIND                   :   VARIOUS_FORM                  (add)
# good                      0.44   :   shape                     0.35
# commonality               0.36   :   regularity                0.28
# similarity                0.34   :   nature                    0.27
# sameness                  0.34   :   consistency               0.26
# ABSTRACTNESS              0.33   :   ABSTRACTNESS              0.26
#
# LARGE_QUANTITY                   :   GREAT_MAJORITY                (add)
# quantity                  0.83   :   majority                  0.73
# size                      0.51   :   good                      0.59
# volume                    0.37   :   minority                  0.45
# quality                   0.36   :   pride                     0.33
# substantiality            0.35   :   continuity                0.29
#
# EFFECTIVE_WAY                    :   EFFICIENT_USE                 (add)
# EFFECTIVENESS             0.44   :   EFFECTIVENESS             0.34
# good                      0.36   :   capability                0.31
# ability                   0.29   :   tractability              0.29
# NECESSITY                 0.28   :   utility                   0.29
# difference                0.27   :   NECESSITY                 0.28
#
# VAST_AMOUNT                      :   HIGH_PRICE                    (add)
# QUANTITY                  0.50   :   price                     0.82
# size                      0.42   :   quality                   0.34
# complexity                0.35   :   volume                    0.30
# substantiality            0.33   :   QUANTITY                  0.29
# nature                    0.33   :   temperature               0.28
#
# OLD_PERSON                       :   ELDERLY_LADY                  (add)
# AGE                       0.43   :   KINDNESS                  0.28
# being                     0.23   :   AGE                       0.25
# KINDNESS                  0.22   :   health                    0.25
# connection                0.22   :   abstemiousness            0.24
# commonness                0.20   :   fertility                 0.22
#
# GENERAL_PRINCIPLE                :   BASIC_RULE                    (add)
# FAIRNESS                  0.40   :   MORALITY                  0.36
# MORALITY                  0.36   :   MANDATE                   0.36
# NECESSITY                 0.36   :   FAIRNESS                  0.35
# MANDATE                   0.33   :   dispensability            0.35
# majority                  0.32   :   NECESSITY                 0.31
#
# EFFECTIVE_WAY                    :   EFFICIENT_USE                 (add)
# EFFECTIVENESS             0.44   :   EFFECTIVENESS             0.34
# good                      0.36   :   capability                0.31
# ability                   0.29   :   tractability              0.29
# NECESSITY                 0.28   :   utility                   0.29
# difference                0.27   :   NECESSITY                 0.28
#
# FEDERAL_ASSEMBLY                 :   NATIONAL_GOVERNMENT           (add)
# MANDATE                   0.28   :   MANDATE                   0.36
# majority                  0.25   :   health                    0.31
# seniority                 0.24   :   independence              0.30
# MINORITY                  0.22   :   crisis                    0.28
# function                  0.22   :   MINORITY                  0.28
#
# LARGE_QUANTITY                   :   GREAT_MAJORITY                (add)
# quantity                  0.83   :   majority                  0.73
# size                      0.51   :   good                      0.59
# volume                    0.37   :   minority                  0.45
# quality                   0.36   :   pride                     0.33
# substantiality            0.35   :   continuity                0.29
#
# CERTAIN_CIRCUMSTANCE             :   PARTICULAR_CASE               (add)
# timing                    0.37   :   substantiality            0.36
# nature                    0.35   :   typicality                0.36
# COMMONNESS                0.32   :   seriousness               0.36
# finality                  0.32   :   fairness                  0.34
# corruptness               0.31   :   COMMONNESS                0.34
#
# EARLIER_WORK                     :   EARLY_STAGE                   (add)
# being                     0.24   :   timing                    0.27
# difficulty                0.21   :   age                       0.26
# good                      0.21   :   maturity                  0.24
# seniority                 0.20   :   height                    0.22
# thoughtfulness            0.20   :   potential                 0.21
#
# DIFFERENT_KIND                   :   VARIOUS_FORM                  (add)
# good                      0.44   :   shape                     0.35
# commonality               0.36   :   regularity                0.28
# similarity                0.34   :   nature                    0.27
# sameness                  0.34   :   consistency               0.26
# ABSTRACTNESS              0.33   :   ABSTRACTNESS              0.26
#
# CENTRAL_AUTHORITY                :   LOCAL_OFFICE                  (add)
# MANDATE                   0.41   :   health                    0.23
# power                     0.38   :   commerce                  0.22
# RESPONSIBILITY            0.35   :   MANDATE                   0.22
# legality                  0.31   :   RESPONSIBILITY            0.19
# FUNCTION                  0.29   :   FUNCTION                  0.19
#
# NEW_LIFE                         :   EARLY_AGE                     (add)
# happiness                 0.34   :   age                       0.84
# humanness                 0.34   :   maturity                  0.36
# domesticity               0.32   :   height                    0.35
# freedom                   0.31   :   dormancy                  0.26
# reality                   0.30   :   susceptibility            0.25
#
# POLITICAL_ACTION                 :   ECONOMIC_DEVELOPMENT          (add)
# action                    0.70   :   commerce                  0.41
# cowardice                 0.32   :   CRISIS                    0.31
# CRISIS                    0.32   :   health                    0.29
# morality                  0.31   :   potential                 0.27
# civility                  0.31   :   modernity                 0.25
#
# FEDERAL_ASSEMBLY                 :   NATIONAL_GOVERNMENT           (add)
# MANDATE                   0.28   :   MANDATE                   0.36
# majority                  0.25   :   health                    0.31
# seniority                 0.24   :   independence              0.30
# MINORITY                  0.22   :   crisis                    0.28
# function                  0.22   :   MINORITY                  0.28
#
# NEW_LIFE                         :   ECONOMIC_DEVELOPMENT          (add)
# happiness                 0.34   :   commerce                  0.41
# humanness                 0.34   :   crisis                    0.31
# domesticity               0.32   :   health                    0.29
# freedom                   0.31   :   potential                 0.27
# reality                   0.30   :   modernity                 0.25
#
# SMALL_HOUSE                      :   LITTLE_ROOM                   (add)
# size                      0.35   :   comfort                   0.33
# majority                  0.25   :   good                      0.29
# quantity                  0.23   :   mind                      0.26
# minority                  0.23   :   actuality                 0.25
# ordinariness              0.19   :   light                     0.24
#
# NEW_BODY                         :   WHOLE_SYSTEM                  (add)
# weight                    0.31   :   capability                0.32
# shape                     0.26   :   integrity                 0.32
# mind                      0.26   :   complexity                0.29
# fullness                  0.25   :   FUNCTION                  0.29
# FUNCTION                  0.24   :   fairness                  0.27
#
# SPECIAL_CIRCUMSTANCE             :   PARTICULAR_CASE               (add)
# timing                    0.29   :   substantiality            0.36
# significance              0.28   :   typicality                0.36
# nature                    0.27   :   seriousness               0.36
# magnitude                 0.26   :   fairness                  0.34
# auspiciousness            0.26   :   commonness                0.34
#
# BETTER_JOB                       :   GOOD_EFFECT                   (add)
# GOOD                      0.56   :   GOOD                      0.67
# position                  0.38   :   difference                0.41
# SHAPE                     0.29   :   SHAPE                     0.37
# consistency               0.29   :   effectiveness             0.32
# responsibility            0.29   :   strength                  0.30
#
# IMPORTANT_PART                   :   SIGNIFICANT_ROLE              (add)
# IMPORTANCE                0.49   :   SIGNIFICANCE              0.41
# SIGNIFICANCE              0.42   :   IMPORTANCE                0.40
# good                      0.37   :   position                  0.39
# necessity                 0.37   :   RESPONSIBILITY            0.39
# RESPONSIBILITY            0.34   :   potential                 0.36
#
# NEW_LAW                          :   BASIC_RULE                    (add)
# MANDATE                   0.36   :   MORALITY                  0.36
# legality                  0.30   :   MANDATE                   0.36
# lawfulness                0.29   :   fairness                  0.35
# MORALITY                  0.26   :   dispensability            0.35
# measure                   0.26   :   necessity                 0.31
#
# NEW_SITUATION                    :   PRESENT_POSITION              (add)
# crisis                    0.42   :   POSITION                  0.80
# POSITION                  0.39   :   presence                  0.34
# possibility               0.34   :   status                    0.33
# reality                   0.34   :   stature                   0.32
# SHAPE                     0.30   :   SHAPE                     0.31
#
# EARLIER_WORK                     :   EARLY_STAGE                   (add)
# being                     0.24   :   timing                    0.27
# difficulty                0.21   :   age                       0.26
# good                      0.21   :   maturity                  0.24
# seniority                 0.20   :   height                    0.22
# thoughtfulness            0.20   :   potential                 0.21
#
# EFFECTIVE_WAY                    :   EFFICIENT_USE                 (add)
# EFFECTIVENESS             0.44   :   EFFECTIVENESS             0.34
# good                      0.36   :   capability                0.31
# ability                   0.29   :   tractability              0.29
# NECESSITY                 0.28   :   utility                   0.29
# difference                0.27   :   NECESSITY                 0.28
#
# NEW_LIFE                         :   EARLY_AGE                     (add)
# happiness                 0.34   :   age                       0.84
# humanness                 0.34   :   maturity                  0.36
# domesticity               0.32   :   height                    0.35
# freedom                   0.31   :   dormancy                  0.26
# reality                   0.30   :   susceptibility            0.25
#
# LARGE_QUANTITY                   :   GREAT_MAJORITY                (add)
# quantity                  0.83   :   majority                  0.73
# size                      0.51   :   good                      0.59
# volume                    0.37   :   minority                  0.45
# quality                   0.36   :   pride                     0.33
# substantiality            0.35   :   continuity                0.29
# """
#
#
# lines = re.split("\n", complete_human_ratings_string)
# # lines = re.split("\n", test)
# # print(complete_human_ratings_string)
#
# seen_examples = []
# examples = {}          #dict mit beispiel : (counter, [5 zeilen mit attributen])
#
#
#
#
#
#
#
#
# number_examples = 0
#
# for i in range(0, len(lines)):
#     # print(line)
#     # print(re.match(regex,line))
#     if re.match(regex,lines[i]):
#         line = lines[i][:-37].strip()
#         number_examples += 1
#         if line not in seen_examples:
#             seen_examples.append(line)
#             examples[line] = [1, lines[i+1:i+6]]
#         else:
#             examples[line][0] += 1
#
# print(number_examples)
# print(len(lines) / 7)
# print(len(list(examples)))
#
# for key in list(examples):
#     print(key, examples[key][0])
#     for line in examples[key][1]:
#         print(line)
#     print()



