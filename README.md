# Home-Credit

跟着北理的大佬参加了Kaggle数据竞赛中Home Credit default risk信贷违约分析比赛。我主要负责对前三张表application_train,previous_application和
bureau的部分特征做数据预处理，主要采用了groupby技术和one-hot编码，对于缺失值采用中位数的方法填充。然后根据我们处理完的三张表的完整数据，利用xgboost，random forest和adaboost三种模型来寻找重要的特征。并且将不重要的变量去除。两个py文件就是我写的数据预处理部分。三张表特征重要性文件夹里面的ipynb文件就是三种模型的代码。

提交的测试数据以cv计算auc值为0.799，我们小组最终取得的成绩在7198支队伍中排313名(前5%)：

![Image text](https://github.com/James0231/Home-Credit/blob/master/img_folder/%E6%88%90%E7%BB%A9.png)


