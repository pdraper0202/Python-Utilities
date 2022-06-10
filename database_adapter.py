import numpy as np
import pandas as pd
import psycopg2
import snowflake.connector


class DatabaseAdapter:
    def __init__(self, database_connection_parameters):
        self._database_connection_parameters = database_connection_parameters
        self.connection = None

    def _connect_to_database(self):
        raise NotImplementedError

    def _disconnect_from_database(self):
        if self.connection is not None:
            self.connection.close()
            self.connection = None

    def pull_from_database(self, sql_file, params=None):
        self._connect_to_database()
        if params is None:
            params = {}
        cursor = self.connection.cursor()
        sqlfile = open(sql_file, 'r')
        statement = sqlfile.read().format(**params)
        cursor.execute(statement)

        try:
            arr = np.array(cursor.fetchall())
            colnames = [d[0] for d in cursor.description]
            data = pd.DataFrame(arr)
            if arr.size !=0:
                data.columns = colnames
        except Exception as e:
            print('An unexpected error occured:', str(e))
        finally:
            sqlfile.close()
            cursor.close()
            self._disconnect_from_database()

        return data

    def run_update_statement(self, sql_file, params=None):
        self._connect_to_database()
        if params is None:
            params = {}
        cursor = self.connection.cursor()
        sqlfile = open(sql_file, 'r')
        statement = sqlfile.read().format(**params)
        cursor.execute(statement)

        try:
            self.connection.commit()
            count = cursor.rowcount
        except Exception as e:
            print('An unexpected error occured:', str(e))
        finally:
            sqlfile.close()
            cursor.close()
            self._disconnect_from_database()

        return count

    def push_to_database(self, df, database_table, number_batches):
        self._connect_to_database()
        cursor = self.connection.cursor()
        columns = ",".join(df.columns.to_list())
        insert_values_list = self._dataframe_to_insert_statements(df, number_batches)
        try:
            for values in insert_values_list:
                cursor.execute(f"INSERT INTO {database_table} ({columns}) VALUES {values}")
                self.connection.commit()
        except Exception as e:
                print('An unexpected error occured:', str(e))
        finally:
            cursor.close()
            self._disconnect_from_database()

    def _dataframe_to_insert_statements(self, df, number_batches=1):
        res=[]
        arr = df.to_numpy().astype('str')
        for bat in list(self._split(range(arr.shape[0]), number_batches)):
            sub_arr = arr[bat, :]
            arg = str(tuple(map(tuple, sub_arr)))[1:-1].rstrip(',').replace("'None'", "NULL")
            res.append(arg)
        return res

    def _split(self, a, n):
        q, r = divmod(len(a), n)
        return (a[i * q + min(i, r):(i + 1) * q + min(i + 1, r)] for i in range(n))

class PostgresAdapter(DatabaseAdapter):
    def __init__(self, database_connection_parameters):
        super().__init__(database_connection_parameters)

    def _connect_to_database(self):
        self.connection = psycopg2.connect(**self._database_connection_parameters)

class SnowflakeAdapter(DatabaseAdapter):
    def __init__(self, database_connection_parameters):
        super().__init__(database_connection_parameters)

    def _connect_to_database(self):
        self.connection = snowflake.connector.connect(**self._database_connection_parameters)
