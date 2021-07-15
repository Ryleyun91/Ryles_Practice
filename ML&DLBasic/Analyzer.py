import pandas as pd
import pymysql

class MarketDB:
    def __init__(self):
        """생성자: MariaDB 연결 및 종목코드 딕셔너리 생성"""
        self.conn = pymysql.connect(host='127.0.0.1', user='root',
                                    password='yoon1424', db='investar', charset='utf8')

        with self.conn.cursor() as curs:
            sql_com = "SELECT * FROM company_info"
            sql_pri = "SELECT * FROM daily_price"
            df_com = pd.read_sql(sql_com, self.conn)
            self.data = pd.read_sql(sql_pri, self.conn)
            curs.execute(sql_com)
            curs.execute(sql_pri)
        self.conn.commit()

        self.com_code = {x: y for x, y in zip(df_com["company"], df_com["code"])}
        self.data["date"] = self.data["date"].apply(str)

    def get_daily_price(self, code, start_date, end_date):
        try:
            int(code)
            return self.data[(self.data["code"] == code) & (start_date <= self.data["date"]) & (self.data["date"] <= end_date)].reset_index()

        except:
            return self.data[(self.data["code"] == self.com_code[code]) & (start_date <= self.data["date"]) & (self.data["date"] <= end_date)].reset_index()