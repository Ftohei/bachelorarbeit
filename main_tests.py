# -*- coding: utf-8 -*-

from  gensim.models import Word2Vec
import numpy as np
import example_reader

#liste möglicher Faktoren: Projektion, Distanzmetrik
# Todo: euklidische Distanz gibt immer dasselbe zurück

core_attributes = ['AGE', 'DIRECTION', 'SIZE', 'SPEED', 'TEMPERATURE', 'COLOR', 'DURATION', 'SMELL', 'TASTE', 'WEIGHT']



def find_next_neighbours(adj, noun, attr, model, top_n = 10, projection_mode ='add'):
    # print adj,noun,attr, projection_mode
    # print "%s %s %s %s" % (projection_mode, adj, noun, attr)
    # print "Berechne nächste Nachbarn für projection mode %s mit %s und %s -> %s" % (projection_mode, adj, noun, attr)
    print "Berechne nächste Nachbarn für projection mode '%s' mit Adjektiv '%s' und Nomen '%s'. Gesuchtes Attribut: '%s'" % (projection_mode, adj, noun, attr)
    result_cosine_sim = []
    result_euclidean = []

    word_list = model.vocab.keys()

    projected_vec = 0

    if projection_mode == 'add':
        projected_vec = np.add(model[adj], model[noun])
    elif projection_mode == 'mult':
        projected_vec = np.multiply(model[adj],  model[noun])

    # print "überprüfe insgesamt %d wörter auf nachbarschaft" % len(word_list)
    i = 0
    for word in word_list:
        cosine_sim = cos_sim(projected_vec,model[word])
        euclid = euclidean_dist(projected_vec,model[word])
        result_cosine_sim.append((word,cosine_sim))
        result_euclidean.append((word,euclid))
        i+=1
        if i % 1000000 == 0:
            print 'iteriere...'
            # print "Iteration %d -> cosine_sim = %f, euclidean = %f" % (i,cosine_sim,euclid)

    #sortiere nach der kosinus-ähnlichkeit absteigend.
    result_cosine_sim.sort(key = lambda x:x[1], reverse=True)
    result_euclidean.sort(key = lambda x:x[1], reverse=True)


    #finde das erste auftreten vom attribut in den wörtern
    index_of_attr_cosine_sim = -1
    i = 0
    for w,cosine_sim in result_cosine_sim:
        if w == attr:
            index_of_attr_cosine_sim = i
            break
        i+=1

    index_of_attr_euclidean = -1
    i = 0
    for w,euclidean_d in result_euclidean:
        if w == attr:
            index_of_attr_euclidean = i
            break
        i+=1

    return result_cosine_sim[0:top_n], index_of_attr_cosine_sim, result_euclidean[0:top_n], index_of_attr_euclidean

def cos_sim(vec1,vec2):
    d1 = np.shape(vec1)[0]
    d2 = np.shape(vec2)[0]
    if d1 != d2:
        print "Ungleiche Dims bei Cos Sim!"

    enumerator = 0.0
    sum1_under_root = 0.0
    sum2_under_root = 0.0

    for i in range(0,d1):
        enumerator += vec1[i] * vec2[i]
        sum1_under_root += vec1[i]**2
        sum2_under_root += vec2[i]**2

    result = enumerator / (np.sqrt(sum1_under_root) * np.sqrt(sum2_under_root))
    return result

def euclidean_dist(vec1,vec2):
    d1 = np.shape(vec1)[0]
    d2 = np.shape(vec2)[0]
    if d1 != d2:
        print "Ungleiche Dims bei Euklidean Dist!"

    sum = 0

    for i in range(0,d1):
        sum += (vec1[i] + vec2[i]) ** 2

    return np.sqrt(float(sum))

#[[attr,adj,noun],[...]...]
aan_list = example_reader.read_attr_adj_noun('/Users/Fabian/Documents/Uni/6. Semester/Bachelorarbeit/Data/beispiele_ATTR_adj_nomen_short.txt')

print 'Lese Liste von Adj-Noun-Attr'
print aan_list
# print u"nächster nachbar"

print "Lade word-embeddings..."
vector_space = Word2Vec.load_word2vec_format('/Users/Fabian/Documents/Uni/6. Semester/Bachelorarbeit/code/GoogleNews-vectors-negative300.bin', binary=True)
print "fertig geladen"




for aan in aan_list:
    for projection_mode in ['add','mult']:
        attr = aan[0]
        adj = aan[1]
        noun = aan[2]

        try:
            top_n_results_cs, index_of_attr_cs, top_n_results_ed, index_of_attr_ed = find_next_neighbours(adj,noun,attr,vector_space,projection_mode=projection_mode)

            string = "%s (%s) %s -> %s\nTop N resultate Kosinus-Ähnlichkeit = %s\nTop N resultate Euklid = %s\nAttribut steht nach Kosinus-Ähnlichkeit auf Platz %d und nach Euklid auf Platz %d" % (adj,projection_mode,noun,attr,top_n_results_cs,top_n_results_ed,index_of_attr_cs,index_of_attr_ed)
            print string
            example_reader.write_to_file('/Users/Fabian/Documents/Uni/6. Semester/Bachelorarbeit/Data/resultate_erste_projektionen.txt',string)

        except KeyError as e:
            print "Wort nicht als Word-embedding gefunden!"
            example_reader.write_to_file('/Users/Fabian/Documents/Uni/6. Semester/Bachelorarbeit/Data/resultate_erste_projektionen.txt',"Wort nicht als Word-embedding gefunden!")
            print e

