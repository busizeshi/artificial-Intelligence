"""
oob
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

X, y = make_moons(n_samples=5000, noise=0.3, random_state=42)

rf_clf = RandomForestClassifier(n_estimators=500, oob_score=True, n_jobs=-1, random_state=42)
rf_clf.fit(X, y)

# 3. 获取 OOB 分数
print(f"OOB Score: {rf_clf.oob_score_:.4f}")

# 对比测试集分数
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf_clf.fit(X_train, y_train)
print(f"Test Set Score: {rf_clf.score(X_test, y_test):.4f}")
