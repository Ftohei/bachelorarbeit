import numpy as np
import theano.tensor as T
from theano import pp
from theano import function
from keras.layers import Input
from keras.models import Model
from gensim.models import Word2Vec
from nltk import word_tokenize
import sys
import file_util
import matplotlib.pyplot as plt


import composition_learning
from subset_experiments import *


all= """
INTROSPECTIVENESS
PROLIXITY
CRITICALITY
ABILITY
CUBICITY
ABSORBENCY
PROPRIETY
INTROVERSION
INTRUSIVENESS
CURLINESS
PURITY
ABSTEMIOUSNESS
CURRENTNESS
QUALITY
INWARDNESS
ABSTRACTNESS
CYCLICITY
QUANTITY
ACCURACY
KINDNESS
ACQUISITIVENESS
LAWFULNESS
DEGREE
READINESS
ACTION
REALITY
DEHISCENCE
LEGALITY
ACTIVENESS
LENGTH
REASONABLENESS
DEPTH
ACTUALITY
DESTRUCTIBILITY
LIGHT
REASSURANCE
RECOGNITION
ADEQUACY
DIFFERENCE
LIKELIHOOD
DIFFICULTY
REGULARITY
LIKENESS
ADMISSIBILITY
REPULSION
LITERACY
AFFECTEDNESS
DIRECTION
REPUTE
DIRECTNESS
LIVELINESS
AGE
AIRWORTHINESS
RESPONSIBILITY
DISPENSABILITY
LOGICALITY
RIGHTNESS
DISPOSITION
ALARM
LOYALTY
LUMINOSITY
DISTANCE
SAMENESS
AMBITION
SARCASM
ANCESTRY
DIVERSENESS
MAGNETISM
DOMESTICITY
ANIMATENESS
SEAWORTHINESS
MAGNITUDE
ANIMATION
MAJORITY
DORMANCY
SENIORITY
APPETIZINGNESS
DRAMA
MALEFICENCE
SENSATIONALISM
SENSITIVITY
APPROPRIATENESS
MALIGNITY
DULLNESS DURATION
MANDATE
SENTIENCE
ASSURANCE
SEPARATION
EASE
MATERIALITY
ASTRINGENCY
SERIOUSNESS
ATTENTION
EFFECTIVENESS
MATURITY
EFFICACY
MEASURE
ATTENTIVENESS
SEX
SHAPE
EMOTIONALITY
MIND
ATTRACTIVENESS
EQUALITY
SHARPNESS
MINDFULNESS
ATTRIBUTION
AUDIBILITY
SIGNIFICANCE
MINORITY
ESSENTIALITY
MODERATION
AUSPICIOUSNESS
EVENNESS
SIMILARITY
AWARENESS
SINCERITY
MODERNITY
EVIL
MODESTY
SIZE
EXCITEMENT
BEAUTY
EXPLICITNESS
MORALITY
BEING
SMELL
SOCIABILITY
EXTINCTION
BENEFICENCE
MOTHERLINESS
MOTION
BENIGNITY
SOCIALITY
FAIRNESS
SOLIDITY
BOLDNESS
MUSICALITY
FAMILIARITY
BREAKABLENESS
FATHERLINESS
NASTINESS
SPEED
NATURALNESS
FEAR
STALENESS
CAPABILITY
NATURE
FELICITY
STATURE
CAREFULNESS
NECESSITY
FERTILITY
CERTAINTY
STATUS
STRENGTH
CHANGEABLENESS
FIDELITY
NICENESS
FINALITY
SUBSTANTIALITY
CHEERFULNESS
NOBILITY
NORMALITY
SUCCESS
CIVILITY
FOREIGNNESS
SUFFICIENCY
FORMALITY
NUMERACY
CLARITY
NUMEROUSNESS
SUSCEPTIBILITY
CLEANNESS
FREEDOM
OBEDIENCE
FRESHNESS
CLEARNESS
TAMENESS TASTE
FRIENDLINESS
OBVIOUSNESS
COLOUR
TEMPERATURE
COMFORT
OFFENSIVENESS
FULLNESS
OPACITY
TEXTURE
FUNCTION
COMMERCE
THICKNESS
GENERALITY
COMMONALITY
ORDINARINESS
ORIGINALITY
COMMONNESS
THOUGHTFULNESS
GENEROSITY
ORTHODOXY
GLUTTONY
TIMIDITY
COMPLETENESS
OTHERNESS
TIMING
GOOD
COMPLEXION
GREGARIOUSNESS
OUTWARDNESS
COMPLEXITY
TRACTABILITY
COMPREHENSIVENESS
HANDINESS
TRUTH
PASSIVITY
CONCRETENESS
TYPICALITY
HAPPINESS
PERFECTION
CONFIDENCE
ULTIMACY
HARDNESS
PERMANENCE
PERMISSIVENESS
CONNECTION
HEALTH
UNFAMILIARITY
USUALNESS
PIETY
HEIGHT
CONSISTENCY
PITCH
CONSPICUOUSNESS
HOLINESS
UTILITY
PLAYFULNESS
VALENCE
CONSTANCY
HONESTY
CONTINUITY
HONORABLENESS
PLEASANTNESS
VIRGINITY
VIRTUE
HUMANENESS
CONVENIENCE
POLITENESS
POPULARITY
VOLUME
CONVENTIONALITY
HUMANNESS
POSITION
CONVERTIBILITY
WARINESS
HUMILITY
WEIGHT
CORRECTNESS
IMMEDIACY
POSSIBILITY
POTENCY
IMPORTANCE
CORRUPTNESS
WETNESS WIDTH
POTENTIAL
COURAGE
INDEPENDENCE
INDIVIDUALITY
POWER
COURTESY
WILDNESS
WORTHINESS
COWARDICE
PRACTICALITY
INTEGRITY
CREATIVITY
INTELLIGENCE
PRESENCE
INTENTIONALITY
PRICE
CREDIBILITY
CRISIS
INTEREST
PRIDE
"""



# print(len(word_tokenize(test2)))

dict = {}


for name, attr_set in [ALL_ATTRIBUTES,CORE_ATTRIBUTES,SELECTED_ATTRIBUTES,MEASUREABLE_ATTRIBUTES,PROPERTY_ATTRIBUTES,WEBCHILD_ATTRIBUTES]:
    dict[name] = {}
    for name2, attr_set2 in [ALL_ATTRIBUTES,CORE_ATTRIBUTES,SELECTED_ATTRIBUTES,MEASUREABLE_ATTRIBUTES,PROPERTY_ATTRIBUTES,WEBCHILD_ATTRIBUTES]:
        dict[name][name2] = len([attr for attr in attr_set if attr in attr_set2 or attr.lower() in ['color','colour']])
        # dict[name][1][name2] = [attr for attr in attr_set if attr not in attr_set2]
        # print([attr for attr in attr_set if attr not in attr_set2])
    # print(name, dict[name][0],"\n\t",dict[name][1])


reihenfolge = ['ALL','CORE','SELECTED','MEASUREABLE','PROPERTY','WEBCHILD']

def pptable(dict):
    tex_string = """\\begin{table}\n\\begin{tabular}{rcccccc}\n\\hline\n"""

    #header
    for i in range(0,len(reihenfolge) - 1):
        tex_string += reihenfolge[i] + " & "
    tex_string += reihenfolge[5] + "\\\\\n"



    for i in range(0,len(reihenfolge)):
        tex_string += reihenfolge[i] + " & "
        for j in range(0,len(reihenfolge) - 1):
            tex_string += str(dict[reihenfolge[i]][reihenfolge[j]]) + " & "
        tex_string += str(dict[reihenfolge[i]][reihenfolge[5]]) + "\\\\\n"

    tex_string += "\\end{tabular}\n\\end{table}"

    print(tex_string)

# pptable(dict)


name, attr_set = ALL_ATTRIBUTES
name2, attr_set2 = WEBCHILD_ATTRIBUTES
print(len(attr_set2))
print(len([attr for attr in attr_set2 if attr not in attr_set]))



#
# for key in list(dict):
#     tex_string += key + "&"
#     for key2 in list(dict[key]):
#         tex_string += key
#
#




#     \begin{tabular}{rlc}
# 	\hline
# & Addition & Similarity\\
# \hline
# 1 & \textbf{age} & 0.25\\
# 2 & potential & 0.21\\
# 3 & success & 0.20\\
# 4 & excitement & 0.19\\
# 5 & maturity & 0.19\\
# 6 & ease & 0.19\\
# 7 & possibility & 0.18\\
# 8 & good & 0.16\\
# 9 & size & 0.16\\
# 10 & reality & 0.15\\
# \end{tabular}





# test2 = sorted(word_tokenize(test2))
# print(test2)
# print(len(test2))
#
# test3 = ""
# for word in test2:
#     test3 += word + "\n"
#
# print(test3)
# HEIPLAS_DEV_SET_PATH = '/Users/Fabian/Documents/Uni/6. Semester/Bachelorarbeit/Data/HeiPLAS-release/HeiPLAS-dev.txt'
# HEIPLAS_TEST_SET_PATH = '/Users/Fabian/Documents/Uni/6. Semester/Bachelorarbeit/Data/HeiPLAS-release/HeiPLAS-test.txt'
#
#
# aan_test, attrs_test, bla, bla = file_util.read_attr_adj_noun(HEIPLAS_TEST_SET_PATH)
# aan_dev, attrs_dev, bla, bla = file_util.read_attr_adj_noun(HEIPLAS_DEV_SET_PATH)
#
# print("Dev set: {} Tripel, {} verschiedene Attribute".format(len(aan_dev),len(attrs_dev)))
# print("Test set: {} Tripel, {} verschiedene Attribute".format(len(aan_test),len(attrs_test)))
#
# print(attrs_test, attrs_dev)
#
# shared_attrs = [attr for attr in attrs_dev if attr in attrs_test]
# print("# gemeinsame Attribute: {}".format(len(shared_attrs)))



# sys.stdout = open('testfile.txt', 'w')
# print("test")

# matr = plt.imshow(np.random.random((20,1)), cmap='Greys_r')
# plt.savefig("test.png")