from sklearn.model_selection import train_test_split
import pandas as pd
import argparse
import numpy as np

RND_SEED = 241


def get_splits_from_df(df):
    """
  Splits manual curation df into train, val and test splits
  RND_SEED is set for reproducibility
  """
    df_certain = df[df['curation_score'].isin([0, 1])]
    assert df_certain['id'].count() == df_certain['id'].nunique()
    np.random.seed(RND_SEED)
    sent_ids = df_certain['id'].unique()
    sent_ids_train, sent_ids_val_test = train_test_split(sent_ids,
                                                         test_size=0.3,
                                                         random_state=RND_SEED)
    sent_ids_val, sent_ids_test = train_test_split(sent_ids_val_test,
                                                   test_size=0.5,
                                                   random_state=RND_SEED)

    train_df = df_certain[df_certain['id'].isin(sent_ids_train)]
    val_df = df_certain[df_certain['id'].isin(sent_ids_val)]
    test_df = df_certain[df_certain['id'].isin(sent_ids_test)]
    return train_df, val_df, test_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--input-file',
                        type=str,
                        default='data/manual_classifications.csv',
                        help='Location of the input file')
    parser.add_argument('--train-file',
                        type=str,
                        default='data/train_paper_classif.csv',
                        help='Location where the training file will be saved')
    parser.add_argument('--val-file',
                        type=str,
                        default='data/val_paper_classif.csv',
                        help='Location where the val file will be saved')
    parser.add_argument('--test-file',
                        type=str,
                        default='data/test_paper_classif.csv',
                        help='Location where the test file will be saved')

    args, _ = parser.parse_known_args()
    print(f'args={args}')

    df = pd.read_csv(args.input_file)
    train_df, val_df, test_df = get_splits_from_df(df)
    train_df.to_csv(args.train_file, index=False)
    val_df.to_csv(args.val_file, index=False)
    test_df.to_csv(args.test_file, index=False)
