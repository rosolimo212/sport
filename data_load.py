import yaml, json, boto3, botocore
import psycopg2 as ps
import pandas as pd
import numpy as np

yaml_file = 'config.yaml'

def read_yaml_config(yaml_file: str, section: str) -> dict:
    with open(yaml_file, 'r') as yaml_stream:
        descriptor = yaml.full_load(yaml_stream)
        if section in descriptor:
            configuration = descriptor[section]
            return configuration
        else:
            logging.error(f"Section {section} not find in the file '{yaml_file}'")
            sys.exit(1)
            
            
def get_data(query: str, file=yaml_file, section='postgres_cloud') -> list:
    settings = read_yaml_config(file, section)
    conn = None
    try:
        conn = ps.connect(**settings)
        cur = conn.cursor()
        cur.execute(query)
        try:
            rows = cur.fetchall()
            colnames = [desc[0] for desc in cur.description]
            df = pd.DataFrame(rows, columns=colnames)
        except (Exception, ps.DatabaseError) as err:
            print(f"PostgreSQL can't execute query - {err}")
            df=pd.DataFrame()
        cur.close()
        conn.close()
        return df
    except (Exception, ps.DatabaseError) as err:
        print(f"PostgreSQL can't execute query - {err}")
    finally:
        if conn is not None:
            conn.close()

def get_engine(file, section='postgres_cloud'):
    settings = read_yaml_config(file, section)
    from sqlalchemy import create_engine
    postgresql_engine_st = "postgresql://"+settings['user']+":"+settings['password']+"@"+settings['host']+"/"+settings['database']
    postgresql_engine = create_engine(postgresql_engine_st)

    return postgresql_engine
            