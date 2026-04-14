# AKShare 说明

根据官方的说明文档，调取 A 股数据的接口函数为：

```python
stock_zh_a_hist(symbol: str = '000001', period: str = 'daily', start_date: str = '19700101', end_date: str = '20500101', adjust: str = '', timeout: float = None) -> pandas.DataFrame
```

具体参数调用说明如下：

```txt
    东方财富网-行情首页-沪深京 A 股-每日行情
    https://quote.eastmoney.com/concept/sh603777.html?from=classic
    :param symbol: 股票代码
    :type symbol: str
    :param period: choice of {'daily', 'weekly', 'monthly'}
    :type period: str
    :param start_date: 开始日期
    :type start_date: str
    :param end_date: 结束日期
    :type end_date: str
    :param adjust: choice of {"qfq": "前复权", "hfq": "后复权", "": "不复权"}
    :type adjust: str
    :param timeout: choice of None or a positive float number
    :type timeout: float
    :return: 每日行情
    :rtype: pandas.DataFrame
```

相应的列元素类型如下

```txt

['日期', '股票代码', '开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌幅', '涨跌额', '换手率']
日期       object
股票代码        str
开盘      float64
收盘      float64
最高      float64
最低      float64
成交量       int64
成交额     float64
振幅      float64
涨跌幅     float64
涨跌额     float64
换手率     float64
dtype: object
```
