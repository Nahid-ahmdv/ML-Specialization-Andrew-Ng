""" Utilities for RecSysNN assigment """
from collections import defaultdict
import csv
import numpy as np
from numpy import genfromtxt
import pickle
import tabulate


def load_data():
    ''' called to load preprepared data for the lab '''
    item_train = genfromtxt('./data/content_item_train.csv', delimiter=',')
    user_train = genfromtxt('./data/content_user_train.csv', delimiter=',')
    y_train    = genfromtxt('./data/content_y_train.csv', delimiter=',')
    with open('./data/content_item_train_header.txt', newline='') as f:    #csv reader handles quoted strings better
        item_features = list(csv.reader(f))[0]
    with open('./data/content_user_train_header.txt', newline='') as f:
        user_features = list(csv.reader(f))[0]
    item_vecs = genfromtxt('./data/content_item_vecs.csv', delimiter=',')

    movie_dict = defaultdict(dict)
    count = 0
#    with open('./data/movies.csv', newline='') as csvfile:
    with open('./data/content_movie_list.csv', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        for line in reader:
            if count == 0:
                count += 1  #skip header
                #print(line) print
            else:
                count += 1
                movie_id = int(line[0])
                movie_dict[movie_id]["title"] = line[1]
                movie_dict[movie_id]["genres"] = line[2]

    with open('./data/content_user_to_genre.pickle', 'rb') as f:
        user_to_genre = pickle.load(f)

    return(item_train, user_train, y_train, item_features, user_features, item_vecs, movie_dict, user_to_genre)

def pprint_train(x_train, features, vs, u_s, maxcount=5, user=True):
    '''
    x_train: This is the input dataset. It is likely a 2D NumPy array where rows correspond to individual data points (e.g., ratings, features, etc.), and columns correspond to different features.
    features: A list of feature names or descriptions that will be used as headers in the output table.
    vs: A value representing the index where the feature list (features) is split into "user-related" and "item-related" features.
    u_s: The number of "user-related" features. This is likely the number of user-specific columns (like user IDs, ratings, etc.).
    maxcount: The maximum number of rows to display from x_train. By default, it's set to 5, meaning only the first 5 rows of data will be shown.
    user: A boolean flag that indicates whether you are working with user-specific features or item-specific features. This influences the format of the output table.
    '''
    """ Prints user_train or item_train nicely """
    if user:
        flist = [".0f", ".0f", ".1f",
                 ".1f", ".1f", ".1f", ".1f", ".1f", ".1f", ".1f", ".1f", ".1f", ".1f", ".1f", ".1f", ".1f", ".1f"]
    else:
        flist = [".0f", ".0f", ".1f", 
                 ".0f", ".0f", ".0f", ".0f", ".0f", ".0f", ".0f", ".0f", ".0f", ".0f", ".0f", ".0f", ".0f", ".0f"]
    '''
    flist: This list specifies how the floating-point numbers should be formatted when printing the table. The formatting is different depending on whether you are printing user-related data or item-related data.
    .0f: This means the number should be formatted as an integer.
    .1f: This means the number should have one decimal place.
    '''
    #Header Setup:
    head = features[:vs]
    if vs < u_s: print("error, vector start {vs} should be greater than user start {u_s}")
    for i in range(u_s):
        head[i] = "[" + head[i] + "]"
    genres = features[vs:]
    hdr = head + genres
    '''
    head: The first vs elements of features are assigned to head. These correspond to the user-related features.
    genres: The remaining features after the index vs are assigned to genres. These correspond to the item-related features.
    hdr: The combined header of both head and genres, which will be used as the column names in the table.
    '''
    #Table Data Construction:
    disp = [split_str(hdr, 5)]
    count = 0
    for i in range(0, x_train.shape[0]):
        if count == maxcount: break
        count += 1
        disp.append([x_train[i, 0].astype(int),
                     x_train[i, 1].astype(int),
                     x_train[i, 2].astype(float),
                     *x_train[i, 3:].astype(float)
                    ])
    '''
    disp: A list that will contain the rows of the table.
    split_str(hdr, 5): This seems to be a function call to split the header string, possibly for formatting purposes. It's unclear without seeing the function split_str.
    For Loop: Iterates through each row in x_train:
    If count exceeds maxcount, the loop stops.
    Otherwise, the row is processed, and the values are converted to their appropriate data types (integer or float).
    The values for each feature are appended as a row to disp.
    '''
    table = tabulate.tabulate(disp, tablefmt='html', headers="firstrow", floatfmt=flist, numalign='center')
    '''
    tabulate.tabulate(): This function converts the data in disp into a nicely formatted table in HTML format.
    tablefmt='html': Specifies that the output should be in HTML format.
    headers="firstrow": The first row of disp (which contains the headers) will be used as the table headers.
    floatfmt=flist: Applies the formatting from flist to the floating-point numbers.
    numalign='center': Centers the numerical values in the table cells.
    '''
    return table


def split_str(ifeatures, smax):
    ''' split the feature name strings to tables fit '''
    ofeatures = []
    for s in ifeatures:
        if not ' ' in s:  # skip string that already have a space
            if len(s) > smax:
                mid = int(len(s)/2)
                s = s[:mid] + " " + s[mid:]
        ofeatures.append(s)
    return ofeatures
    

def print_pred_movies(y_p, item, movie_dict, maxcount=10):
    """ print results of prediction of a new user. inputs are expected to be in
        sorted order, unscaled. """
    count = 0
    disp = [["y_p", "movie id", "rating ave", "title", "genres"]]

    for i in range(0, y_p.shape[0]):
        if count == maxcount:
            break
        count += 1
        movie_id = item[i, 0].astype(int)
        disp.append([np.around(y_p[i, 0], 1), item[i, 0].astype(int), np.around(item[i, 2].astype(float), 1),
                     movie_dict[movie_id]['title'], movie_dict[movie_id]['genres']])

    table = tabulate.tabulate(disp, tablefmt='html', headers="firstrow")
    return table

def gen_user_vecs(user_vec, num_items):
    """ given a user vector return:
        user predict maxtrix to match the size of item_vecs """
    user_vecs = np.tile(user_vec, (num_items, 1))
    return user_vecs

# predict on  everything, filter on print/use
def predict_uservec(user_vecs, item_vecs, model, u_s, i_s, scaler):
    """ given a scaled user vector, does the prediction on all movies in scaled print_item_vecs returns
        an array predictions sorted by predicted rating,
        arrays of user and item, sorted by predicted rating sorting index
    """
    y_p = model.predict([user_vecs[:, u_s:], item_vecs[:, i_s:]])
    y_pu = scaler.inverse_transform(y_p)

    if np.any(y_pu < 0):
        print("Error, expected all positive predictions")
    sorted_index = np.argsort(-y_pu, axis=0).reshape(-1).tolist()  #negate to get largest rating first
    sorted_ypu   = y_pu[sorted_index]
    sorted_items = item_vecs[sorted_index]
    sorted_user  = user_vecs[sorted_index]
    return(sorted_index, sorted_ypu, sorted_items, sorted_user)
                
def get_user_vecs(user_id, user_train, item_vecs, user_to_genre):
    """ given a user_id, return:
        user train/predict matrix to match the size of item_vecs
        y vector with ratings for all rated movies and 0 for others of size item_vecs """

    if not user_id in user_to_genre:
        print("error: unknown user id")
        return None
    else:
        user_vec_found = False
        for i in range(len(user_train)):
            if user_train[i, 0] == user_id:
                user_vec = user_train[i]
                user_vec_found = True
                break
        if not user_vec_found:
            print("error in get_user_vecs, did not find uid in user_train")
        num_items = len(item_vecs)
        user_vecs = np.tile(user_vec, (num_items, 1))

        y = np.zeros(num_items)
        for i in range(num_items):  # walk through movies in item_vecs and get the movies, see if user has rated them
            movie_id = item_vecs[i, 0]
            if movie_id in user_to_genre[user_id]['movies']:
                rating = user_to_genre[user_id]['movies'][movie_id]
            else:
                rating = 0
            y[i] = rating
    return(user_vecs, y)

def get_item_genres(item_gvec, genre_features):
    ''' takes in the item's genre vector and list of genre names
    returns the feature names where gvec was 1 '''
    offsets = np.nonzero(item_gvec)[0]
    genres = [genre_features[i] for i in offsets]
    return genres


def print_existing_user(y_p, y, user, items, ivs, uvs, movie_dict, maxcount=10):
    """ print results of prediction for a user who was in the database.
        Inputs are expected to be in sorted order, unscaled.
    """
    count = 0
    disp = [["y_p", "y", "user", "user genre ave", "movie rating ave", "movie id", "title", "genres"]]
    count = 0
    for i in range(0, y.shape[0]):
        if y[i, 0] != 0:  # zero means not rated
            if count == maxcount:
                break
            count += 1
            movie_id = items[i, 0].astype(int)

            offsets = np.nonzero(items[i, ivs:] == 1)[0]
            genre_ratings = user[i, uvs + offsets]
            disp.append([y_p[i, 0], y[i, 0],
                         user[i, 0].astype(int),      # userid
                         np.array2string(genre_ratings, 
                                         formatter={'float_kind':lambda x: "%.1f" % x},
                                         separator=',', suppress_small=True),
                         items[i, 2].astype(float),    # movie average rating
                         movie_id,
                         movie_dict[movie_id]['title'],
                         movie_dict[movie_id]['genres']])

    table = tabulate.tabulate(disp, tablefmt='html', headers="firstrow", floatfmt=[".1f", ".1f", ".0f", ".2f", ".1f"])
    return table
