from load_data import *
from train import *
from predict import *
import pickle
from params import *
from datetime import datetime
from keras import backend as K

def max_k(l, k):
    if len(l) < k: return 0.0
    pivot = l[-1]
    right = [pivot] + [x for x in l[:-1] if x >= pivot]
    right_len = len(right)
    if right_len == k:
        return pivot
    if right_len > k:
        return max_k(right, k)
    else:
        left = [x for x in l[:-1] if x < pivot]
        return max_k(left, k - right_len)

def read_data():
    if params['dataset'] == 'CiteULike-a':
        docs = []
        file = '../data/citeulike/ctrsr_datasets/citeulike-a/raw-data.csv'
        f = open(file, 'rb')
        reader = csv.reader(f)
        #nltk.download()
        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        for line in reader:
            doc_id, title, abstract = line[0], line[3], line[4]
            try:
                int(doc_id)
                if doc_id != '12479':
                    content = title + '. ' + abstract
                else:
                    content = title
                content = content.decode('ISO-8859-1')
                sentences = tokenizer.tokenize(content)

                for i in range(len(sentences)):
                    text = sentences[i]
                    text = [w.lower() for w in re.split('\W', text) if w]
                    text = [word.lower() for word in text]
                    # porter_stemmer = PorterStemmer()
                    # text = [porter_stemmer.stem(word) for word in text]

                    sentences[i] = text
                    # print sentences[i]
                doc = []
                for sen in sentences:
                    doc += sen
                docs.append(doc)
                print(docs)
            except ValueError:
                pass

f = open('test_set', 'rb')
test_set = pickle.load(f)
f.close()
f = open('doc2content', 'rb')
doc2content = pickle.load(f)
f.close()

n_epoch = params['n_epoch']
batch_size = params['batch_size']
n_users = 5551
n_items = 16980 + 1

#model = load_model('../MODELS/2017-07-06 15:56:03.795058/my_model_0.00592090867309_.h5')

print('building model')
num_docs, max_doc_len = params['num_docs'], params['max_doc_len']
model = build(n_users,n_items,max_doc_len,num_docs)
print('finished building model')
f = open('word2id', 'rb')
word2id = pickle.load(f)
f.close()

try:
    model.load_weights('../data/model_weights')

    # training_set = load_training_data()
    # generator = batch_generator_train(batch_size, training_set, doc2content)
    # intermediate_layer_model = Model(input=model.input, output=model.layers[12].output)
    # print training_set

    get_layer_output = K.function([model.get_layer(name='pos_doc_input').input, K.learning_phase()], [model.get_layer(name='activation_1').output])
    batches = len(doc2content) // params['batch_size']
    doc2content_len = len(doc2content)
    batches_doc2content_len = (batches + 1) * params['batch_size']

    if not os.path.exists(os.path.dirname('../result/doc_res_0')):
        try:
            os.makedirs(os.path.dirname('../result/doc_res_0'))
        except:
            print 'create path failed'

    x = []
    res_output = open('../result/doc_res_0', 'w')

    for i in range(batches_doc2content_len):
        if i % params['batch_size'] == 0 and i != 0:
            x = np.asarray(x, 'int32')
            layer_output = get_layer_output([x, 0])[0]
            #print(layer_output)

            for j in range(params['batch_size']):
                #print("doc_id: ", i + j - params['batch_size'])
                kth = max_k(layer_output[j], 21)
                print kth
                res = []

                #print('doc_id: ', str(i + j - params['batch_size']))
                for k in range(len(x[j])):
                    word_id = x[j][k]
                    if word_id in word2id.values():
                        word = list(word2id.keys())[list(word2id.values()).index(word_id)]
                        if layer_output[j][k] >= kth:
                            word = word.upper()
                        #print(word, ' ', end='')
                        res.append(word)
                if i + j - params['batch_size'] < doc2content_len:
                    res_str = 'doc_id: ' + str(i + j - params['batch_size']) + '\n'
                    for k in range(len(res)):
                        res_str += str(res[k]) + ' '
                    res_str += '\n\n'
                    print res_str
                    res_output.write(res_str)

                # for k in range(len(layer_output[j])):
                #     if layer_output[j][k] >= kth:
                #         word_id = x[j][k]
                #         if word_id in word2id.values():
                #             word = list(word2id.keys())[list(word2id.values()).index(word_id)]
                #             res.append((word, layer_output[j][k]))
                #             #print(word, layer_output[j][k], ' ')
                #
                # #res.sort(key=lambda x: x[1], reverse=True)
                # doc_id = i + j - params['batch_size']
                # if doc_id < doc2content_len:
                #     res_str = 'doc_id: ' + str(doc_id) + '\n' + str(res) + '\n\n'
                #     res_output.write(res_str)
                #     print(res_str)
                # print("\n")
            x = []

        if i < doc2content_len:
            x.append(doc2content[i])
        else:
            x.append(doc2content[0])

    #print(len(x))
    res_output.close()

    # cnt = 0
    # for x, y in generator:
    #     cnt += 1
    #     print(cnt)
    #     if cnt > 1:
    #         break
    #
    #     pos_docs = x[1]
    #     pos_target_docs = x[3]
    #    # print(pos_target_docs)
    #
    #     layer_output = intermediate_layer_model.predict(x)
    #     for i in range(params['batch_size']):
    #         print("doc_id: ", pos_docs[i])
    #         k = max_k(layer_output[i], 20)
    #
    #         for j in range(len(layer_output[i])):
    #             #dic = {}
    #             if layer_output[i][j] > k:
    #                 word_id = pos_target_docs[i][j]
    #                 if word_id in word2id.values():
    #                     word = list(word2id.keys())[list(word2id.values()).index(word_id)]
    #                     dic[layer_output[i][j]] = word
    #                     print word, layer_output[i][j], ' ',
    #             #sorted(dic.keys())
    #             #print dic
    #         print("\n")

    #model = load_model('../MODELS/2017-06-30 23:43:10.178530/my_model_0.319809730519_.h5')
    '''
    model.load_weights('../data/weights')

    #get_layer_output = K.function([model.layers[0].input, K.learning_phase()], [model.layers[16].output])
    training_set = load_training_data()
    generator = batch_generator_train(batch_size, training_set, doc2content)

    #layer_name = 'my_layer'
    intermediate_layer_model = Model(input=model.input, output=model.layers[12].output)
    #intermediate_output = intermediate_layer_model.predict(data

    layer_outputs = []
    cnt = 0
    for x, y in generator:
        cnt += 1
        if cnt > 1:
            break;
        pos_target_docs = x[3]
        print pos_target_docs
        #layer_output = get_layer_output([x, 1])[0]
        layer_output = intermediate_layer_model.predict(x)
        for i in xrange(params['batch_size']):
            #print x[i]
            #print layer_output[i]
            max = 0.0
            id = 0
            for j in xrange(len(layer_output[i])):
                if layer_output[i][j] > max:
                    id = j
                    max = layer_output[i][j]
            #print id, ": ", layer_output[i][id]
            word = pos_target_docs[i][id]
            print word

        #layer_outputs.append(layer_output)

    #print layer_outputs
    '''
    '''
    model = Train(model=model,n_epoch=n_epoch,batch_size=batch_size,test_set=test_set)
    recall_50 = Test(model, doc2content, test_set)
    print 'Saving model'
    filepath = '../MODELS/' + str(datetime.now()) + '/'
    filename = 'my_model_' + str(recall_50) + '_.h5'
    if not os.path.exists(os.path.dirname(filepath + filename)):
        try:
            os.makedirs(os.path.dirname(filepath + filename))
        except:
            print 'create path failed'

    model.save(filepath + filename)
    f = open(filepath + 'params.txt','wb')
    f.write(str(params))
    f.close()
    '''

except KeyboardInterrupt:
    pass
