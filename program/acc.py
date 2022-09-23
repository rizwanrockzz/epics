import numpy as np

filename = input("Enter a file name(@Desktop)：")

with open("/Users/kaze/Desktop/" + filename) as f:
    maintxt = [s.strip() for s in f.readlines()]
    # print(maintxt)

data_dic = {}
result_dic = {}

for i in range( int(len(maintxt)/3) ):
    i = 3 * i

    title = maintxt[i].split(" ")[2]
    acc = float(maintxt[i+2].split("accuracy:")[1])

    if not title in data_dic:
        data_dic[title] = []
        result_dic[title] = []
    else: pass

    data_dic[title].append(acc)


for name in data_dic:
    acc = np.array(data_dic[name])
    result_dic[name].append(str(len(acc))+"Data for each session")
    result_dic[name].append(np.mean(acc))
    result_dic[name].append(np.max(acc))
    result_dic[name].append(np.min(acc))
    result_dic[name].append(np.std(acc))

    print(name)
    print(result_dic[name])
