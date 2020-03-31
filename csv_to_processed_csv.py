import pandas as pd


def create_path(df, base_path):

    df['path'] = df.apply(lambda x: base_path+"aligned/"+x['user_id']+"/landmark_aligned_face.%s.%s"
                                                                      %(x['face_id'], x['original_image']), axis=1)

def filter_df(df):

    dict_age = {'(0, 2)' : 0,
                '(4, 6)' : 1,
                '(8, 12)' : 2,
                '(15, 20)' : 3,
                '(25, 32)' : 4,
                '(38, 43)' : 5,
                '(48, 53)' : 6,
                '(60, 100)' : 7}


    df['f'] = df.age.apply(lambda x: int(x in dict_age))
    df = df[df.f == 1]
    return df


base_path = "./Dataset-copy/"

dict_age = {'(0, 2)' : 0,
            '(4, 6)' : 1,
            '(8, 12)' : 2,
            '(15, 20)' : 3,
            '(25, 32)' : 4,
            '(38, 43)' : 5,
            '(48, 53)' : 6,
            '(60, 100)' : 7}

bag = 3

all_indexes = list(range(5))

accuracies = []

#for test_id in (all_indexes):
test_id = 2
train_id = [j for j in all_indexes if j!=test_id]
print(train_id, test_id)

train_df = pd.concat([pd.read_csv(base_path+"fold_%s_data.txt"%i, sep="\t") for i in train_id])
test_df = pd.read_csv(base_path+"fold_%s_data.txt"%test_id, sep="\t")

train_df = filter_df(train_df)
test_df = filter_df(test_df)

print(train_df.shape, test_df.shape)

train_df = create_path(train_df, base_path=base_path)
test_df = create_path(test_df, base_path=base_path)
# train_df.to_csv("train_processed.csv","w")
# test_df.to_csv("test_processed.csv", "w")
train_df.()
