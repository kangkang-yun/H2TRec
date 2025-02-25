
import pickle

dataset = 'yelp'


"""
构造宽模型所需要的txt文件
"""

with open('./'+dataset+'/dataset_filter5.pkl', 'rb') as f:
    train_set = pickle.load(f)
    valid_set = pickle.load(f)
    test_set = pickle.load(f)

with open('./'+dataset+'/list_filter5.pkl', 'rb') as f:
    u_items_list = pickle.load(f)
    u_users_list = pickle.load(f)
    u_users_items_list = pickle.load(f)
    i_users_list = pickle.load(f)
    (user_count, item_count, rate_count) = pickle.load(f)
    u_fans_list = pickle.load(f)
    u_fans_items_list = pickle.load(f)

f_trust = open('./'+dataset+'/trust_data.txt', "w")
for truster, trustee in enumerate(u_users_list):
    if trustee[0] == 0:
        continue
    for j in trustee:
        context = '{} {} 1'.format(truster, j)
        f_trust.write('\n')
        f_trust.write(context)
f_trust.close()

f_test = open('./'+dataset+'/new_test_set_filter5.txt', "w")
for i in test_set:
    context = '{} {} {}'.format(i[0], i[1], i[2])
    f_test.write('\n')
    f_test.write(context)
f_test.close()

f_train = open('./'+dataset+'/new_train_set_filter5.txt', "w")
for i in train_set:
    context = '{} {} {}'.format(i[0], i[1], i[2])
    f_train.write('\n')
    f_train.write(context)
f_train.close()

f_valid = open('./'+dataset+'/new_valid_set_filter5.txt', "w")
for i in valid_set:
    context = '{} {} {}'.format(i[0], i[1], i[2])
    f_valid.write('\n')
    f_valid.write(context)
f_valid.close()

print("finished!!")
