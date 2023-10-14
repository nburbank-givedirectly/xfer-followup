import os
from databricks import sql
import hashlib
import pandas as pd
from pandas import read_sql
import sqlparse
from nb_secrets import ACCESS_TOKEN


SERVER_HOSTNAME = "dbc-6e84555a-4b45.cloud.databricks.com"
HTTP_PATH = "/sql/1.0/warehouses/d48c4e74e9fb7034"


def generate_hash(string):
    hash_object = hashlib.md5(string.encode())
    digest = hash_object.hexdigest()
    return digest[:10]

def write_string_to_file(file_path, string_to_write):
    with open(file_path, 'w') as file:
        file.write(string_to_write)

def get_df(query, limit=1000, print_q=False):
    if limit != 0:
        query = query.strip() + f" limit {limit}"

    query = sqlparse.format(query.strip(), reindent=True, keyword_case='upper')
    hash_key = generate_hash(query.lower())
    if print_q:
        print(query)

    file_path = f"data_cache/{hash_key}.pq"
    query_path =f"data_cache/{hash_key}.sql"

    if os.path.exists(file_path):
        print(f"Loading {file_path}")
        df = pd.read_parquet(file_path)
    else:
        print(f"The file {file_path} does not exist.")

        df = None
        with sql.connect(
            server_hostname=SERVER_HOSTNAME,
            http_path=HTTP_PATH,
            access_token=ACCESS_TOKEN,
        ) as connection:
            df = read_sql(query, connection)
        write_string_to_file(query_path,query)
        df.to_parquet(file_path)
        

    return df
