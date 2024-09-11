settings = {
    'api_key': 'eb5a249e88e6f45184e66384474fa048',
    'series_ids': [
        'FEDFUNDS', 'GDP', 'CPIAUCSL', 'CUSR0000SAH1', 'CPILFESL', 'PCE', 
        'PRFI', 'PNFI', 'EXPGS', 'HOUST', 'DSPI', 'DGS2', 'DGS5', 'DGS10', 
        'AAA', 'BAA', 'WTISPLC', 'IMPGS', 'GCE', 'FGCE', 'GDPCTPI', 
        'PCEPI', 'PCEPILFE', 'PAYEMS', 'UNRATE', 'INDPRO', 'CUMFNS','M1','M1V','M2',
        'VIXCLS', 'BAMLH0A0HYM2', 'BAA10Y', 'BAMLC0A4CBBBEY', 'BAMLC0A1CAAAEY', 'T10YIE', 'T5YIE', 'DGS3MO', 'DGS30', 'BAA10YM', 'MORTGAGE30US', 'TIPS5', 'RSXFS', 'TCU', 'UMCSENT', 'NETEXP', 'PI', 'PPIACO',
        'M1REAL','M2REAL',
    ],
    'start_date': '2003-01-01',
    'end_date': '2024-05-01',
    'frequency_map': {
        'FEDFUNDS': 'm',
        'GDP': 'q',
        'CPIAUCSL': 'm',
        'CUSR0000SAH1': 'm',
        'CPILFESL': 'm',
        'PCE': 'm',
        'PRFI': 'q',
        'PNFI': 'q',
        'EXPGS': 'q',
        'HOUST': 'm',
        'DSPI': 'm',
        'DGS2': 'd',
        'DGS5': 'd',
        'DGS10': 'd',
        'AAA': 'm',
        'BAA': 'm',
        'WTISPLC': 'm',
        'IMPGS': 'q',
        'GCE': 'q',
        'FGCE': 'q',
        'GDPCTPI': 'q',
        'PCEPI': 'm',
        'PCEPILFE': 'm',
        'BSHCFAW': 'm',
        'PAYEMS': 'm',
        'UNRATE': 'm',
        'INDPRO': 'm',
        'CUMFNS': 'm',
        # New Features (More Financial's Oriented)
        'M1' : 'm', # M1 Money Stock
        'M1V' : 'q', # M1 Velocity
        'M2' : 'm', # M2 Money Stock
        'VIXCLS': 'd', # CBOE Volatility Index
        'BAMLH0A0HYM2': 'd', # ICE BofA US High Yield Index Option-Adjusted Spread
        'BAA10Y': 'd', # Moody's Seasoned Baa Corporate Bond Yield Relative to Yield on 10-Year Treasury Constant Maturity
        'BAMLC0A4CBBBEY': 'd', # ICE BofA US Corporate BBB Option-Adjusted Spread
        'BAMLC0A1CAAAEY': 'd', # ICE BofA US Corporate AAA Option-Adjusted Spread
        'T10YIE': 'd', # 10-Year Breakeven Inflation Rate
        'T5YIE' : 'd', # 5-Year Breakeven Inflation Rate
        'DGS3MO ' : 'd', # 3-Month Treasury Constant Maturity Rate
        'DGS30 ' : 'd', # 30-Year Treasury Constant Maturity Rate
        'BAA10YM' : 'm', # Moody's Seasoned Baa Corporate Bond Yield
        'MORTGAGE30US' : 'w', # 30-Year Fixed Rate Mortgage Average in the United States
        'TIPS5' : 'd', # 5-Year Treasury Inflation-Indexed Security, Constant Maturity
        'RSXFS' : 'm', # Advance Retail Sales: Retail Trade
        'TCU' : 'm', # Capacity Utilization Total Index
        'UMCSENT' : 'm', # University of Michigan: Consumer Sentiment
        'NETEXP' : 'q', # Net Exports of Goods and Services
        'PI' : 'm', # Personal Income
        'PPIACO': 'm', # Producer Price Index by Commodity: All Commodities
        'M1REAL' : 'm', # Real M1 Money Stock
        'M2REAL' : 'm', # Real M2 Money Stock

    },
    'dagshub_repo' : 'najibabounasr/MacroEconomicAPI',

}