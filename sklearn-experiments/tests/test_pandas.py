import unittest
import numpy as np
import pandas as pd


class TestDataFrame(unittest.TestCase):
    def test_data(self):
        data = {'col1': [1, 2], 'col2': [3, 4]}
        df = pd.DataFrame(data=data)
        self.assertEqual(df.shape, (2, 2))
        self.assertEqual(df.dtypes['col1'], 'int64')
        self.assertEqual(df.dtypes['col2'], 'int64')
        np.testing.assert_array_equal(df['col1'], np.array([1, 2]))
        np.testing.assert_array_equal(df['col2'], np.array([3, 4]))
        np.testing.assert_array_equal(df.columns.values, np.array(['col1', 'col2']))
        np.testing.assert_array_equal(df.index.values, np.array([0, 1]))

    def test_index_and_columns(self):
        df = pd.DataFrame([[150, 50], [160, 60], [170, 70]],
                          index=['student1', 'student2', 'student3'],
                          columns=['height', 'weight'])
        self.assertEqual(df.shape, (3, 2))
        np.testing.assert_array_equal(df.columns.values, np.array(['height', 'weight']))
        np.testing.assert_array_equal(df.index.values, np.array(['student1', 'student2', 'student3']))

    def test_dtypes(self):
        df = pd.DataFrame({'col1': [1.0, 2.0] * 3,
                           'col2': ['A', 'B'] * 3,
                           'col3': [True, False] * 3})
        self.assertEqual(df.dtypes['col1'], 'float64')
        self.assertEqual(df.dtypes['col2'], 'object')
        self.assertEqual(df.dtypes['col3'], 'bool')
        df = df.astype({'col1': 'float32'})
        self.assertEqual(df.dtypes['col1'], 'float32')

    def test_select_dtypes(self):
        df = pd.DataFrame({'col1': [1.0, 2.0] * 3,
                           'col2': ['A', 'B'] * 3,
                           'col3': [True, False] * 3})
        df_1 = df.select_dtypes(include=['object', 'bool'])
        np.testing.assert_array_equal(df_1.columns.values, np.array(['col2', 'col3']))
        df_2 = df.select_dtypes(exclude=['bool'])
        np.testing.assert_array_equal(df_2.columns.values, np.array(['col1', 'col2']))

    def test_loc(self):
        # Get/Set operations for a group of rows and columns
        df = pd.DataFrame([[150, 50], [160, 60], [170, 70]],
                          index=['student1', 'student2', 'student3'],
                          columns=['height', 'weight'])
        df_1 = df.loc[['student1', 'student3']]
        df_2 = df.loc['student1':'student3']
        df_3 = df.loc[[True, False, True]]
        df_4 = df.loc[df['weight'] >= 60]

        np.testing.assert_array_equal(df_1.index.values, np.array(['student1', 'student3']))
        np.testing.assert_array_equal(df_2.index.values, np.array(['student1', 'student2', 'student3']))
        np.testing.assert_array_equal(df_3.index.values, np.array(['student1', 'student3']))
        np.testing.assert_array_equal(df_4.index.values, np.array(['student2', 'student3']))

        df_5 = df.loc[['student1'], ['height']]
        df_6 = df.loc[[True, False, True], [True, False]]
        df_7 = df.loc[df['weight'] >= 60, ['height', 'weight']]
        df_8 = df.loc[lambda df: df['weight'] == 60]

        np.testing.assert_array_equal(df_5.index.values, np.array(['student1']))
        np.testing.assert_array_equal(df_5.columns.values, np.array(['height']))
        np.testing.assert_array_equal(df_6.index.values, np.array(['student1', 'student3']))
        np.testing.assert_array_equal(df_6.columns.values, np.array(['height']))
        np.testing.assert_array_equal(df_7.index.values, np.array(['student2', 'student3']))
        np.testing.assert_array_equal(df_7.columns.values, np.array(['height', 'weight']))
        np.testing.assert_array_equal(df_8.index.values, np.array(['student2']))
        np.testing.assert_array_equal(df_8.columns.values, np.array(['height', 'weight']))

    def test_iloc(self):
        # Position-based Get/Set operations for a group of rows and columns
        df = pd.DataFrame({'height': [150, 160, 170],
                           'weight': [50, 60, 70]})
        df_1 = df.iloc[[0, 2]]
        df_2 = df.iloc[:3]
        df_3 = df.iloc[[True, False, True]]

        np.testing.assert_array_equal(df_1.index.values, np.array([0, 2]))
        np.testing.assert_array_equal(df_2.index.values, np.array([0, 1, 2]))
        np.testing.assert_array_equal(df_3.index.values, np.array([0, 2]))

        df_5 = df.iloc[[0], [0]]
        df_6 = df.iloc[[True, False, True], [True, False]]
        df_8 = df.iloc[lambda x: x.index % 2 == 0]

        np.testing.assert_array_equal(df_5.index.values, np.array([0]))
        np.testing.assert_array_equal(df_5.columns.values, np.array(['height']))
        np.testing.assert_array_equal(df_6.index.values, np.array([0, 2]))
        np.testing.assert_array_equal(df_6.columns.values, np.array(['height']))
        np.testing.assert_array_equal(df_8.index.values, np.array([0, 2]))
        np.testing.assert_array_equal(df_8.columns.values, np.array(['height', 'weight']))

    def test_group_and_sort(self):
        df = pd.DataFrame({'col1': [1.0, 2.0] * 3,
                           'col2': ['A', 'B'] * 3,
                           'col3': [True, False] * 3})
        mean_df = df.groupby(['col3']).mean()
        self.assertEqual(mean_df.shape, (2,1))
        np.testing.assert_array_equal(mean_df.columns.values, np.array(['col1']))
        sorted_mean_df = mean_df.sort_values(by=['col1'])
        np.testing.assert_array_equal(sorted_mean_df.index.values, np.array([True, False]))
        np.testing.assert_array_equal(sorted_mean_df['col1'].values, np.array([1.0, 2.0]))

    def test_describe(self):
        df = pd.DataFrame({'col1': [1.0, 2.0] * 3,
                           'col2': ['A', 'B'] * 3,
                           'col3': [True, False] * 3})
        desc = df.describe(include='all')
        np.testing.assert_array_equal(desc.index.values, np.array(['count', 'unique', 'top', 'freq', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']))
        np.testing.assert_array_equal(desc.columns.values, np.array(['col1', 'col2', 'col3']))

    def test_head_and_tail(self):
        df = pd.DataFrame({'col1': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']})
        head = df.head(3)
        np.testing.assert_array_equal(head.values, np.array([['A'], ['B'], ['C']]))
        head = df.tail(3)
        np.testing.assert_array_equal(head.values, np.array([['F'], ['G'], ['H']]))

    def test_insert_and_pop(self):
        df = pd.DataFrame({'height': [150, 160, 170],
                           'weight': [50, 60, 70]})
        np.testing.assert_array_equal(df.columns.values, np.array(['height', 'weight']))
        df.insert(1, 'age', [15, 16, 17])
        np.testing.assert_array_equal(df.columns.values, np.array(['height', 'age', 'weight']))
        df.pop('weight')
        np.testing.assert_array_equal(df.columns.values, np.array(['height', 'age']))

    def test_drop(self):
        df = pd.DataFrame({'age': [15, 16, 17],
                           'height': [150, 160, 170],
                           'weight': [50, 60, 70]})
        df_1 = df.drop(columns=['age'])
        np.testing.assert_array_equal(df_1.columns.values, np.array(['height', 'weight']))
        df_2 = df.drop(index=[1])
        np.testing.assert_array_equal(df_2.index.values, np.array([0, 2]))

    def test_isna_and_notna(self):
        df = pd.DataFrame({'col1': [1, 2, np.NaN],
                           'col2': ['A', None, 'B'],
                           'col3': ['a', '', 'b']})

        df_1 = df.isna()
        np.testing.assert_array_equal(df_1['col1'].values, np.array([False, False, True]))
        np.testing.assert_array_equal(df_1['col2'].values, np.array([False, True, False]))
        np.testing.assert_array_equal(df_1['col3'].values, np.array([False, False, False]))
        df_2 = df.notna()
        np.testing.assert_array_equal(df_2['col1'].values, np.array([True, True, False]))
        np.testing.assert_array_equal(df_2['col2'].values, np.array([True, False, True]))
        np.testing.assert_array_equal(df_2['col3'].values, np.array([True, True, True]))

    def test_dropna_and_fillna(self):
        pass

    def test_apply(self):
        df = pd.DataFrame({'A': [1, 2, 3],
                           'B': [2, 4, 6],
                           'C': [3, 6, 9],
                           'D': [4, 8, 12]
        })
        print('df.shape=',df.shape)
        df_log = df[['A', 'B', 'C']].apply(np.log, axis=1).round(3)
        print(df_log)

    def test_cut_and_qcut(self):
        pass


if __name__ == '__main__':
    unittest.main()
