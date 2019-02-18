import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

ds = pd.read_csv('data/sample-data.csv')

tf = TfidfVectorizer(analyzer = 'word', ngram_range = (1, 3), min_df = 0, stop_words = 'english')
tfidf_matrix = tf.fit_transform(ds['description'])

cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)

results = {}

for idx, row in ds.iterrows():
    similar_indices = cosine_similarities[idx].argsort()[:-100:-1]
    similar_items = [(cosine_similarities[idx][i], ds['id'][i]) for i in similar_indices]

    results[row['id']] = similar_items[1:]

print('Complete.')

# prediction
def item(id):
    return ds.loc[ds['id'] == id]['description'].tolist()[0].split(' - ')[0]

def recommend(item_id, num = 3):
    print('Recommending {0} products similar to {1} ...'.format(num, item(item_id)))
    print('-------')

    recs = results[item_id][:num]

    for rec in recs:
        print('Recommended: {0} (score: {1})' .format(item(rec[1]), rec[0]))


recommend(item_id = 11, num = 5)

# further tests
for i in range(1, 11):
    print('\n')
    recommend(item_id = i)
    print('\n')