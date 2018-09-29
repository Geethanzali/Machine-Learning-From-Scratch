
# coding: utf-8

# In[ ]:


import sys
import csv
import math
import numpy as np

#train_file = sys.argv[1]
#test_file = sys.argv[2]

datafile1 = open(sys.argv[1], 'r')
train_file = csv.reader(datafile1)
train_file.next()

col=[]

for row in train_file:
	c=row[-1]
	col.append(c)
u, indices = np.unique(col, return_index=True)

positive = ["democrat", "A", "y", "before1950", "yes", "morethan3min", "fast", "expensive","high", "Two", "large"]
val = [1 if a in positive else 0 for a in u]
c=val.index(1)
d=val.index(0)

datafile1.close()

datafile1 = open(sys.argv[1], 'r')
train_file = datafile1.readlines()

datafile2 = open(sys.argv[2], 'r')
test_file = datafile2.readlines()

def read_data_from_file(myreader):
	positive = ["democrat", "A", "y", "before1950", "yes", "morethan3min", "fast", "expensive",
	            "high", "Two", "large"]
	#with open(myreader) as file_obj:
		#myreader = file_obj.readlines()
	myreader = [x.strip("\r\n") for x in myreader]
	myreader = [x.strip("\n") for x in myreader]
	myreader = [x.replace(" ", "") for x in myreader]
	attr_name =myreader[0].split(",")
	attr_val = [[1 if a in positive else 0 for a in inst.split(",")] for inst in myreader[1:]]
	return (attr_name, attr_val)

def entropy(Array):
	sum = 0
	for x in Array:
		sum += x
	if sum == 0 or sum == len(Array):
		return 0
	p = float(sum) / len(Array)
	return - p * math.log(p, 2) - (1 - p) * math.log(1 - p, 2)

class Node:
	def __init__(self, entropy, label = 1, attr = -1, l = None, r = None):
		self.entropy = entropy
		self.l = l
		self.r = r
		self.label = label
		self.attr = attr
		self.pos = 0
		self.neg = 0
		self.fattr = -1
		self.mi = 0

def get_best(en, Ex, Attr):
	pool = []
	for attr in Attr:
		pos_temp = []
		neg_temp = []
		for ex in Ex:
			if ex[attr] == 1:
				pos_temp.append(ex[-1])
			else:
				neg_temp.append(ex[-1])
		pool.append(en - len(pos_temp)/float(len(Ex)) * entropy(pos_temp) - len(neg_temp)/float(len(Ex)) * entropy(neg_temp))
	return (max(pool), Attr[pool.index(max(pool))])

def ID3(Examples, Attributes, attr):
	pos = 0
	for ex in Examples:
		pos += ex[-1]
	if pos == 0 or pos == len(Examples):
		root = Node(0)
		root.pos = pos
		root.neg = len(Examples) - pos
		root.attr = attr
		if pos == 0:
			root.label = 0
		elif pos == len(Examples):
			root.label = 1
		return root
	frac = float(pos) / len(Examples)
	root = Node(- frac * math.log(frac, 2) - (1 - frac) * math.log(1 - frac, 2))
	root.pos = pos
	root.neg = len(Examples) - pos
	root.attr = attr
	root.label = 0 if pos <= len(Examples) / 2 else 1

	if len(Attributes) == 0:
		root.label = 0 if pos <= len(Examples)/2 else 1
		return root
	else:
		root.mi, A = get_best(root.entropy, Examples, Attributes)
		root.fattr = A
		new_exl = []
		new_exr = []
		for ex in Examples:
			if ex[A] == 1:
				new_exl.append(ex)
			else:
				new_exr.append(ex)
		if len(new_exl) == 0:
			root.label = 0 if pos <= len(Examples) / 2 else 1
			return root
		if len(new_exr) == 0:
			root.label = 0 if pos <= len(Examples) / 2 else 1
			return root
		new_attr = [attr for attr in Attributes if attr != A]
		if root.mi > 0.0:
			root.l = ID3(new_exl, new_attr, A)
			root.r = ID3(new_exr, new_attr, A)
	return root

def preO(root,l,depth):
	if root == None or depth < 0:
		return
	print_out = ""
	if depth == 4:
		print_out += "[" + str(root.pos) + "+/" + str(root.neg) + "-]"
	else:
		if depth <= 3:
			for i in range(0,2-depth):    
				print_out += "| "
		if l == 1:
			print_out += train_name[root.attr] + " = y: [" + str(root.pos) + "+/" + str(root.neg) + "-]"
		else:
			print_out += train_name[root.attr] + " = n: [" + str(root.pos) + "+/" + str(root.neg) + "-]"
	print print_out
	preO(root.l, 1, depth - 1)
	preO(root.r, 0, depth - 1)

current = []
def get_pred(root, ex,depth):
	global current
	if root == None or depth < 0:
		return
	if ex[root.fattr] == 1:
		current.append(root.label)
		get_pred(root.l, ex, depth - 1)
	else:
		current.append(root.label)
		get_pred(root.r, ex, depth - 1)

error_train = 0.0
train_name, train_val = read_data_from_file(train_file)
attr_num = len(train_name) - 1


depth = int(sys.argv[3])
#depth=3
root = ID3(train_val, [x for x in range(attr_num)], -1)
preO(root, -1, depth)

Val=[]
for ex in train_val:
	positive = ["democrat", "A", "y", "before1950", "yes", "morethan3min", "fast", "expensive",
	            "high", "Two", "large"]
	current = []
	get_pred(root, ex, depth)
	if current[-1] != ex[-1]:
		error_train += 1
	val = [u[c] if current[-1]==1 else u[d]]
	Val.append(val)
k1 = [i[0] for i in Val]
with open(sys.argv[4], 'wb') as out1:
	out1.writelines(["%s\n" % i for i in k1])
	out1.close

error_test = 0.0
test_name, test_val = read_data_from_file(test_file)

Val2=[]
for ex in test_val:
	current = []
	get_pred(root, ex, depth)
	if current[-1] != ex[-1]:
		error_test += 1
	val = [u[c] if current[-1]==1 else u[d]]
	Val2.append(val)
k = [i[0] for i in Val2]
with open(sys.argv[5], 'wb') as out2:
	out2.writelines(["%s\n" % i for i in k])
	out2.close



with open(sys.argv[6], 'wb') as out:
	out.writelines("error(train): " + str(error_train / len(train_val)))
	out.writelines("\nerror(test): " + str(error_test / len(test_val)))
	out.close

