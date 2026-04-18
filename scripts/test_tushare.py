import os

import tushare as ts
from dotenv import load_dotenv

load_dotenv()
ts.set_token(os.environ["TUSHARE_TOKEN"])
pro = ts.pro_api()

# 测试日线数据
df = pro.daily(ts_code="000001.SZ", start_date="20240101", end_date="20240110")
print(df.head())
print("总计:", len(df), "行")
