import pandas as pd

def get_df_info(df: pd.DataFrame):
    index = 0
    for column in df.columns:
        column_name = column
        column_type = str(df[column].dtypes)
        column_values = "(continous)" if column_type == "float64" else df[column].unique()
        column_nulls = df[df[column].isnull()].shape[0]
        print("\nСтолбец {0} (тип {1}) имеет {2} пропусков из {3} значений, {4}% (индекс {5})"
            .format(column_name, column_type, column_nulls, len(df.values), column_nulls/len(df.values) * 100, index, column_values))
        index = index + 1