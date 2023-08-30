import csv
import numpy as np
import matplotlib.pyplot as plt
import random


def load_data(filepath):
    dataset = []
    with open(filepath, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        # only return the first 20 pokemons
        for i in range(0, 20):
            row = reader.__next__()
            # ignore unused columns
            del row['Generation']
            del row['Legendary']
            # convert to ints
            row['#'] = int(row['#'])
            row['Total'] = int(row['Total'])
            row['HP'] = int(row['HP'])
            row['Attack'] = int(row['Attack'])
            row['Defense'] = int(row['Defense'])
            row['Sp. Atk'] = int(row['Sp. Atk'])
            row['Sp. Def'] = int(row['Sp. Def'])
            row['Speed'] = int(row['Speed'])
            dataset.append(row)
    return dataset


def calculate_x_y(stats):
    # takes one pokemon, and calculate its x and y, return in a tuple
    x = stats['Attack'] + stats['Sp. Atk'] + stats['Speed']
    y = stats['Defense'] + stats['Sp. Def'] + stats['HP']
    ans = (x, y)
    return ans


def hac(dataset):
    clusters = []
    for i in range(0, len(dataset)):
        skip = False
        temp = list(dataset[i])
        # skip the entire row of data if any nan or inf exists
        for data in temp:
            if np.isnan(data) or np.isinf(data):
                skip = True
        if not skip:
            temp.append(i)
            clusters.append([temp])         # append the coordinates x, y by their original cluster number
    results = []
    cluster_num = len(clusters)         # track new cluster numbers
    while len(clusters) > 1:
        min_d = np.inf          # record min distance in the current iteration
        merge_a, merge_b, merge_i, merge_j = -1, -1, -1, -1         # reset temp variables, they records best merge
        for i in range(0, len(clusters)):
            for j in range(i + 1, len(clusters)):
                for data_i in clusters[i]:
                    for data_j in clusters[j]:
                        # get distance between any two points in any two clusters
                        distance = np.sqrt((data_i[0] - data_j[0])**2 + (data_i[1] - data_j[1])**2)
                        # take shortest distance, if same, smaller first cluster number, if same again, second number
                        if ((distance < min_d) or (distance == min_d and merge_a > min(data_i[2], data_j[2])) or
                                (distance == min_d and merge_a == min(data_i[2], data_j[2]) and
                                 merge_b > max(data_i[2], data_j[2]))):
                            # record best merge case in the current iteration
                            min_d = distance
                            merge_a = min(data_i[2], data_j[2])
                            merge_b = max(data_i[2], data_j[2])
                            merge_i = i
                            merge_j = j
        results.append([merge_a, merge_b, min_d, len(clusters[merge_i]) + len(clusters[merge_j])])      # record merge
        # merge clusters
        clusters[merge_i].extend(clusters[merge_j])
        for i in range(0, len(clusters[merge_i])):
            clusters[merge_i][i][2] = cluster_num
        cluster_num += 1
        del clusters[merge_j]
    return np.matrix(results)           # return results as NumPy matrix


def random_x_y(m):
    rst = []
    # generate m pairs of [x, y], where 0 < x < 360, 0 < y < 360
    for i in range(0, m):
        rst.append([random.randint(1, 359), random.randint(1, 359)])
    return rst


def imshow_hac(dataset):
    # reused code from hac() to plot the graph
    clusters = []
    x_vals = []
    y_vals = []
    for i in range(0, len(dataset)):
        skip = False
        temp = list(dataset[i])
        for data in temp:
            if np.isnan(data) or np.isinf(data):
                skip = True
        if not skip:
            x_vals.append(temp[0])
            y_vals.append(temp[1])
            temp.append(i)      # cluster number
            temp.append(i)      # original cluster number
            clusters.append([temp])
    plt.scatter(x_vals, y_vals)         # plot original data points
    plt.show()
    cluster_num = len(clusters)         # track new cluster numbers
    while len(clusters) > 1:
        min_d = np.inf          # record min distance in the current iteration
        # reset temp variables, they records best merge
        merge_a, merge_b, merge_i, merge_j, idx_a, idx_b = -1, -1, -1, -1, -1, -1
        for i in range(0, len(clusters)):
            for j in range(i + 1, len(clusters)):
                for data_i in clusters[i]:
                    for data_j in clusters[j]:
                        # get distance between any two points in any two clusters
                        distance = np.sqrt((data_i[0] - data_j[0])**2 + (data_i[1] - data_j[1])**2)
                        # take shortest distance, if same, smaller first cluster number, if same again, second number
                        if ((distance < min_d) or (distance == min_d and merge_a > min(data_i[2], data_j[2])) or
                                (distance == min_d and merge_a == min(data_i[2], data_j[2]) and
                                 merge_b > max(data_i[2], data_j[2]))):
                            # record best merge case in the current iteration
                            min_d = distance
                            merge_a = min(data_i[2], data_j[2])
                            merge_b = max(data_i[2], data_j[2])
                            if merge_a == data_i[2]:
                                idx_a, idx_b = data_i[3], data_j[3]
                            else:
                                idx_b, idx_a = data_i[3], data_j[3]
                            merge_i = i
                            merge_j = j
        # connect two data points that represent the current iteration of HAC, then wait 0.1s to plot the next line
        plt.plot([x_vals[idx_a], x_vals[idx_b]], [y_vals[idx_a], y_vals[idx_b]])
        plt.pause(0.1)
        # merge clusters
        clusters[merge_i].extend(clusters[merge_j])
        for i in range(0, len(clusters[merge_i])):
            clusters[merge_i][i][2] = cluster_num
        cluster_num += 1
        del clusters[merge_j]
