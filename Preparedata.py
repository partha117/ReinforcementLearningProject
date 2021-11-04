import uuid
import random
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def get_combined_full_dataset(project_name, test_percentage=0.4, non_match_size=30, use_uuid=True):
    print("Combining file {}".format(project_name))
    df1 = pd.read_csv("{}.csv".format(project_name), delimiter='\t')
    df2 = pd.read_csv("{}_features.csv".format(project_name))
    df3 = pd.read_csv("{}_features_file_content.csv".format(project_name))
    df4 = pd.merge(df2, df3, left_on='cid', right_on='cid', how='inner')
    df5 = pd.merge(df1, df4, left_on='id', right_on='report_id', how='inner')
    df5['report'] = df5['summary'] + df5['description']
    df5['project_name'] = project_name.split("/")[-1]
    if use_uuid:
        mapper = {item: str(uuid.uuid4()) for item in df5['id'].unique().tolist()}
        for key, value in mapper.items():
            df5.loc[df5['id'] == key, 'id'] = value
    df5.to_csv("{}_complete.csv".format(project_name), index=False)
    train_pos, test_pos = train_test_split(df5[df5['match'] == 1], test_size=test_percentage, random_state=13, shuffle=False)
    train, test = df5[df5['bug_id'].isin(train_pos['bug_id'])], df5[df5['bug_id'].isin(test_pos['bug_id'])]
    test.to_csv("Data/TestData/" + "{}_test.csv".format(project_name.split("/")[-1]), index=False)
    train = train.copy().reset_index(drop=True)
    small_train = pd.DataFrame(columns=train.columns)
    for item in train['bug_id'].unique():
        temp = pd.concat((train[(train['bug_id'] == item) & (train['match'] == 1)],
                          train[(train['bug_id'] == item) & (train['match'] == 0)].head(non_match_size)))
        small_train = pd.concat((small_train, temp))
    small_train.drop(columns=set(small_train.columns) - {'id', 'cid', 'report', 'file_content', 'match'}, inplace=True)
    return small_train


def create_random_dataset(dataset_list, primary_id='id', full_size=1000):
    random.seed(13)
    temp_df = pd.DataFrame(columns=dataset_list[0].columns)
    median_group_size = np.array([item.groupby(primary_id).size().median() for item in dataset_list])
    all_unique_list = np.array([len(item[primary_id].unique().tolist())for item in dataset_list])
    all_unique_list = all_unique_list / np.sum(all_unique_list)
    for item, percentage, group_size in zip(dataset_list, all_unique_list.tolist(), median_group_size):
        all_primary_id = item[primary_id].unique().tolist()
        sample_size = min(round((full_size * percentage) / group_size), len(all_primary_id))
        sampled_primary_id = random.sample(all_primary_id, sample_size)
        cons = item[item[primary_id].isin(sampled_primary_id)]
        temp_df = temp_df.append(item[item[primary_id].isin(sampled_primary_id)]).reset_index(drop=True)
    return temp_df.sample(frac=1, random_state=13).reset_index(drop=True)

if __name__ == "__main__":
    t1 = get_combined_full_dataset(
        "Data/IntermediateData/AspectJ")
    t2 = get_combined_full_dataset(
        "Data/IntermediateData/Tomcat")
    t3 = create_random_dataset([t1, t2], primary_id='id', full_size=5000)
    df1, df2, df3, df4, df5, df6 = get_combined_full_dataset(
        "Data/IntermediateData/Birt"), get_combined_full_dataset(
        "Data/IntermediateData/AspectJ"), get_combined_full_dataset(
        "Data/IntermediateData/Tomcat"), get_combined_full_dataset(
        "Data/IntermediateData/SWT"), get_combined_full_dataset(
        "Data/IntermediateData/JDT"), get_combined_full_dataset(
        "Data/IntermediateData/Eclipse_Platform_UI")
    combined_df = create_random_dataset([df1, df2, df3, df4, df5, df6], primary_id='id', full_size=5000)
    combined_df.to_csv("Data/TrainData/Bench_BLDS_Dataset.csv", index=False)