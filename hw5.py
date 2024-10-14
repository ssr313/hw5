import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
# 加载训练集和测试集
train_data = pd.read_csv('train1_icu_data.csv')
train_labels = pd.read_csv('train1_icu_label.csv', header=None, names=['Survived'])
test_data = pd.read_csv('test1_icu_data.csv')
test_labels = pd.read_csv('test1_icu_label.csv', header=None, names=['Survived'])

# 合并数据和标签
X_train = train_data
y_train = train_labels['Survived']
X_test = test_data
y_test = test_labels['Survived']
# 初始化随机森林分类器
rf = RandomForestClassifier(n_estimators=100, random_state=42)

# 训练模型
rf.fit(X_train, y_train[1:])
# 在测试集上进行预测
y_pred = rf.predict(X_test)

# 计算准确率
accuracy = accuracy_score(y_test[1:], y_pred)
print(f"Accuracy: {accuracy}")

# 打印分类报告
print(classification_report(y_test[1:], y_pred))
# 获取特征重要性
feature_importances = rf.feature_importances_

# 将特征重要性与特征名称结合
features = pd.DataFrame({'Feature': X_train.columns, 'Importance': feature_importances})
print("Top 5 most important features:")
print(features.head(5))
print("\nTop 5 least important features:")
print(features.tail(5))
# 排序并绘制特征重要性图
features.sort_values(by='Importance', ascending=False, inplace=True)
plt.figure(figsize=(10, 8))
plt.barh(features['Feature'], features['Importance'])
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.title('Feature Importance in Random Forest')
plt.savefig('feature_importance.png')
plt.show()