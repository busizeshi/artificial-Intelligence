import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

movies=pd.read_csv("movies.csv")
print(movies.head(10))

movies_ohe=movies.drop('genres', axis=1).join(movies.genres.str.get_dummies())

movies_ohe.set_index(['movieId', 'title'], inplace=True)

# 频繁项集
frequent_itemssets_movies=apriori(movies_ohe, min_support=0.025, use_colnames=True)
print(frequent_itemssets_movies)
print('-----------------------'*5)
# 关联规则
rules_movies=association_rules(frequent_itemssets_movies, metric='lift', min_threshold=1.25)
print(rules_movies)