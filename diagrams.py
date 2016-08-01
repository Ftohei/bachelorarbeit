import matplotlib.pyplot as plt
import numpy as np


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





# These are the "Tableau 20" colors as RGB.
tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]

# Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.
for i in range(len(tableau20)):
    r, g, b = tableau20[i]
    tableau20[i] = (r / 255., g / 255., b / 255.)


models = ['nn_weighted_adjective_noun_identity','nn_weighted_adjective_identity','add','nn_tensor_product_identity','adj','mitchell_lapata_reversed_2']
nice_labels = ['Distinctly weighted Noun and Adjective', 'Weighted Adjective', 'Vector Addition', 'Tensor Product', 'Adjective', 'Dilating the Noun']
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

def print_p1_p5(path, data_string, evaluate_for = 'test', name = ''):

    legend_size = 8

    plt.clf()

    print("--------------------Evaluiere {}".format(evaluate_for).upper())

    result_dict = extract_data_from_string(data_string)

    if evaluate_for == 'test':
        plt.xlabel('Test Sets')
    elif evaluate_for == 'train':
        plt.xlabel('Training Sets')
    else:
        print("Weder Train noch Test spezifiziert!")
        quit(9)

    plt.ylabel('Precision@1')
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
        plt.plot(list(range(0,5)), test_results_p1, '-', c=tableau20[line_type], label=nice_labels[line_type])
        plt.xticks(list(range(0,5)), labels, rotation=15)
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
    plt.ylabel('Precision@5')

    plt.clf()

    if evaluate_for == 'test':
        plt.xlabel('Test Sets')
    elif evaluate_for == 'train':
        plt.xlabel('Training Sets')
    else:
        print("Weder Train noch Test spezifiziert!")
        quit(9)
    plt.ylabel('Precision@5')
    plt.xlim([-0.5,4.5])
    plt.ylim([0,1])

    for i in range(0,5):
        plt.axvline(i, color='0.25',linestyle=':')

    line_type = 0
    for model in models:
        test_results_p5 = [p_5 for trainset,testset, p_1, p_5 in result_dict[model]]
        plt.plot(list(range(0,5)), test_results_p5, '-', c=tableau20[line_type], label=nice_labels[line_type])
        plt.xticks(list(range(0,5)), labels, rotation=15)
        line_type += 1

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
    nice_labls = [nice_labels[0],nice_labels[2]]

    legend_size = 8

    plt.clf()



    f, axarr = plt.subplots(1, 5, figsize=(20  , 6  ))

    # axarr[0,0].set_xlabel('Training Sets')
    axarr[0].set_ylabel('Precision@1')
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
            axarr[i].plot(list(range(0,5)), test_results_p1, '-', c=tableau2[line_type], label=nice_labls[line_type])
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






# print_p1_p5("/Users/Fabian/Documents/Uni/6. Semester/Bachelorarbeit/tex/grafiken_ba",
#             results_one_string_subset_test, 'test')
#
# print_p1_p5("/Users/Fabian/Documents/Uni/6. Semester/Bachelorarbeit/tex/grafiken_ba",
#             results_one_string_subset_train, 'train')
#
#
# print_p1_p5("/Users/Fabian/Documents/Uni/6. Semester/Bachelorarbeit/tex/grafiken_ba",
#             results_one_string_core_test, 'train', name='test_on_core')
#
#
# print_p1_p5("/Users/Fabian/Documents/Uni/6. Semester/Bachelorarbeit/tex/grafiken_ba",
#             results_one_string_selected_test, 'train', name='test_on_selected')
#
# print_p1_p5("/Users/Fabian/Documents/Uni/6. Semester/Bachelorarbeit/tex/grafiken_ba",
#             results_one_string_measureable_test, 'train', name='test_on_measureable')
#
# print_p1_p5("/Users/Fabian/Documents/Uni/6. Semester/Bachelorarbeit/tex/grafiken_ba",
#             results_one_string_property_test, 'train', name='test_on_property')
#
# print_p1_p5("/Users/Fabian/Documents/Uni/6. Semester/Bachelorarbeit/tex/grafiken_ba",
#             results_one_string_webchild_test, 'train', name='test_on_webchild')

print_ax_arr_test_sets("/Users/Fabian/Documents/Uni/6. Semester/Bachelorarbeit/tex/grafiken_ba",
                       [results_one_string_core_test,results_one_string_selected_test,
                        results_one_string_measureable_test,results_one_string_property_test,
                        results_one_string_webchild_test], 'all_train_sets')