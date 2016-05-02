from nltk import word_tokenize
import codecs

def read_attr_adj_noun(filename,encoding='utf-8'):
    result = []
    with codecs.open(filename,encoding='utf-8') as f:
        complete_string = f.read()
        # print complete_string
        tokens = word_tokenize(complete_string)
        # print tokens
        for i in range(0,len(tokens),3):
            aan = []
            for j in range(i,i+3):
                aan.append((tokens[j].lower()).encode('utf-8'))
            # print aan
            if aan[0] == 'direction_orientation':
                aan[0] = 'direction'.encode('utf-8')
            result.append(aan)

    return result

def write_to_file(filename,string,encoding='utf-8'):
    with codecs.open(filename,'a+',encoding='utf-8') as f:
        try:
            f.writelines(string + "\n")
        except UnicodeEncodeError:
            print "Encoding Error: %s" % string
        except UnicodeDecodeError:
            print "Decoding Error: %s" % string
