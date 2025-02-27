# -*- coding: utf-8 -*-
"""
create on Sep 24, 2019

@author: wangshuo
"""

import random
import pickle
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.io import loadmat

# 1234

random.seed(1234)

workdir = './'


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='yelp', help='dataset name: Ciao/Epinions')
parser.add_argument('--test_prop', default=0.1, help='the proportion of data used for test')
args = parser.parse_args()
print(args.dataset)
# load data
if args.dataset == 'Ciao':
	click_f = loadmat('./Ciao/rating.mat')['rating']
	trust_f = loadmat('./Ciao/trustnetwork.mat')['trustnetwork']
elif args.dataset == 'Epinions':
	click_f = np.loadtxt('./Epinions/ratings_data.txt', dtype = np.int32)
	trust_f = np.loadtxt('./Epinions/trust_data.txt', dtype = np.int32)
elif args.dataset == 'yelp':
	click_f = np.loadtxt('./yelp/ratings.txt', dtype=np.int32)
	trust_f = np.loadtxt('./yelp/trust.txt', dtype = np.int32)

click_list = []
trust_list = []

u_items_list = []
u_users_list = []
u_users_items_list = []
u_fans_list = []
u_fans_items_list = []
i_users_list = []

pos_u_items_list = []
pos_i_users_list = []

user_count = 0
item_count = 0
rate_count = 0

for s in click_f:
	uid = s[0]
	iid = s[1]
	if args.dataset == 'Ciao':
		label = s[3]
	elif args.dataset == 'Epinions' or args.dataset == 'yelp':
		label = s[2]


	if uid > user_count:
		user_count = uid
	if iid > item_count:
		item_count = iid
	if label > rate_count:
		rate_count = label
	click_list.append([uid, iid, label])

pos_list = []
for i in range(len(click_list)):
	pos_list.append((click_list[i][0], click_list[i][1], click_list[i][2]))

# remove duplicate items in pos_list because there are some cases where a user may have different rate scores on the same item.
pos_list = list(set(pos_list))

# filter user less than 5 items
pos_df = pd.DataFrame(pos_list, columns=['uid', 'iid', 'label'])
filter_pos_list = []
user_in_set, user_out_set = set(), set()
for u in tqdm(range(user_count + 1)):
	hist = pos_df[pos_df['uid'] == u]   # hist为编号为u的用户的评分记录
	if len(hist) < 5:                   # 过滤掉评分记录项少于5的用户
		user_out_set.add(u)
		continue
	user_in_set.add(u)
	u_items = hist['iid'].tolist()
	u_ratings = hist['label'].tolist()
	filter_pos_list.extend([(u, iid, rating) for iid, rating in zip(u_items, u_ratings)])
print('modelled user numebr and filtering number: ', len(user_in_set), len(user_out_set))
print('data size before and after filtering: ', len(pos_list), len(filter_pos_list))

# train, valid and test data split
print('test prop: ', args.test_prop)
print("=================")
pos_list = filter_pos_list

random.shuffle(pos_list)
num_test = int(len(pos_list) * args.test_prop)
test_set = pos_list[:num_test]
valid_set = pos_list[num_test:2 * num_test]
train_set = pos_list[2 * num_test:]


print('Train samples: {}, Valid samples: {}, Test samples: {}, Total samples: {}'.format(len(train_set), len(valid_set), len(test_set), len(pos_list)))

with open('./'+args.dataset+'/dataset_filter5.pkl', 'wb') as f:
	pickle.dump(train_set, f)
	pickle.dump(valid_set, f)
	pickle.dump(test_set, f)



pos_df = pd.DataFrame(pos_list, columns = ['uid', 'iid', 'label'])
train_df = pd.DataFrame(train_set, columns = ['uid', 'iid', 'label'])
valid_df = pd.DataFrame(valid_set, columns = ['uid', 'iid', 'label'])
test_df = pd.DataFrame(test_set, columns = ['uid', 'iid', 'label'])

click_df = pd.DataFrame(click_list, columns = ['uid', 'iid', 'label'])
train_df = train_df.sort_values(axis = 0, ascending = True, by = 'uid')
pos_df = pos_df.sort_values(axis = 0, ascending = True, by = 'uid')

"""
u_items_list: 存储每个用户交互过的物品iid和对应的评分，没有则为[(0, 0)]
u_items_list: items rated by the user and corresponding rating, if no rated item, u_items_list = [(0,0)]
"""
for u in tqdm(range(user_count + 1)):
	hist = train_df[train_df['uid'] == u]
	u_items = hist['iid'].tolist()
	u_ratings = hist['label'].tolist()
	if u_items == []:
		u_items_list.append([(0, 0)])
	else:
		u_items_list.append([(iid, rating) for iid, rating in zip(u_items, u_ratings)])



train_df = train_df.sort_values(axis = 0, ascending = True, by = 'iid')

"""
i_users_list: 存储与每个物品相关联的用户及其评分，没有则为[(0, 0)]
i_users_list: given an item, rating users and corresponding ratings, if no rating user, i_users_list = [(0,0)]
"""
userful_item_set = set()
for i in tqdm(range(item_count + 1)):
	hist = train_df[train_df['iid'] == i]
	i_users = hist['uid'].tolist()
	i_ratings = hist['label'].tolist()
	if i_users == []:
		i_users_list.append([(0, 0)])
	else:
		i_users_list.append([(uid, rating) for uid, rating in zip(i_users, i_ratings)])
		userful_item_set.add(i)

print('item size before and after filtering: ', item_count, len(userful_item_set))




count_f_origin, count_f_filter = 0,0
for s in trust_f:
	uid = s[0]
	fid = s[1]
	count_f_origin += 1
	if uid > user_count or fid > user_count:
		continue
	if uid in user_out_set or fid in user_out_set:
		continue
	trust_list.append([uid, fid])
	count_f_filter += 1

print('u-u relation filter size changes: ', count_f_origin, count_f_filter)


"""
u_users_list: social friends list with the form [[u1, u2], [u1, u4, u5]] (存储每个用户互动过的用户uid)
u_users_items_list: social friends list with their rated items (存储用户每个朋友的物品iid列表)
"""



trust_network = pd.DataFrame(trust_list, columns = ['uid', 'fid'])       						# uid信任fid
trust_enthusiasm = trust_network.sort_values(axis=0, ascending=True, by='uid')
trust_popularity = trust_network.sort_values(axis=0, ascending=True, by='fid')

count_0, count_1 = 0, 0
for u in tqdm(range(user_count + 1)):
	hist = trust_enthusiasm[trust_enthusiasm['uid'] == u]
	u_users = hist['fid'].unique().tolist()
	if u_users == []:
		u_users_list.append([0])
		u_users_items_list.append([[(0, 0)]])
		# count_0 += 1
	else:
		u_users_list.append(u_users)
		uu_items = []
		for uid in u_users:
			uu_items.append(u_items_list[uid])
		u_users_items_list.append(uu_items)
		# count_1 += 1

for u in tqdm(range(user_count + 1)):
	hist = trust_popularity[trust_popularity['fid'] == u]
	u_users = hist['uid'].unique().tolist()
	if u_users == []:
		u_fans_list.append([0])
		u_fans_items_list.append([[(0, 0)]])
	else:
		u_fans_list.append(u_users)
		uu_items = []
		for uid in u_users:
			uu_items.append(u_items_list[uid])
		u_fans_items_list.append(uu_items)

with open('./'+args.dataset+'/list_filter5.pkl', 'wb') as f:
	pickle.dump(u_items_list, f)				# 用户u交互过的物品items集合
	pickle.dump(u_users_list, f)                # 用户u信任的用户users集合
	pickle.dump(u_users_items_list, f)			# 用户u信任的用户users交互过的物品items集合
	pickle.dump(i_users_list, f)				# 点评过物品i的用户users集合
	pickle.dump((user_count, item_count, rate_count), f)
	pickle.dump(u_fans_list, f)					# 信任用户u的用户fans集合
	pickle.dump(u_fans_items_list, f)			# 信任用户u的用户fans交互过的物品items集合

print("finish preprocessing!")


